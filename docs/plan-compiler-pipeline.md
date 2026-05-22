# Chitin compiler pipeline refactor

The goal is to make Chitin explain every decision it made. One config schema, one resolved build plan, explicit stages, and verification artifacts.

The current smell: `core.py` is 1,457 lines where detection, policy, reconstruction, decomposition, seam repair, environment heuristics, and reporting all happen. The service has drifted independently -- `poisson_depth=8` is hardcoded in three places (`chitin_service/models.py:63`, `app.py:90`, `store.py:202`) while core.py auto-selects depth from point count. That drift should be architecturally impossible.

## Current state — COMPLETE (2026-05-21)

All six phases shipped. core.py went from 1,457 lines to 322. result.py went from 282 to 55. All 58 tests pass.

```
src/chitin/
  core.py          322 lines. Orchestration only: extract, extract_from_arrays,
                    extract_from_mesh, extract_from_rigged_mesh.
  analyze.py       InputAnalysis dataclass + analyze_arrays().
  resolve.py       ResolvedConfig dataclass + resolve_config(). Single policy locus.
  config.py        Frozen dataclass. User-facing knobs only.
  plan.py          BuildPlan: step names + detected dict. Lightweight.
  result.py        Hull, BoneInfo, LodHulls, ExtractionResult (data only, thin delegates to exporters).
  phys.py          .phys reader + validator.
  cli.py           Argparse, dispatch. --bundle/-b flag for artifact bundles.
  gltf_skin.py     GLB binary skin parser.
  preflight.py     Size/format preflight checks.
  convert.py       FBX -> GLB via Blender.
  hooks.py         Post-process hook runner.
  _poisson_worker.py  Subprocess isolation for Open3D.

  adapters/
    __init__.py    SkinData, AdapterResult, load() dispatcher.
    ply.py         PLY adapter (positions, opacity, covariance).
    gltf.py        GLTF/GLB/FBX adapter (skin detection via gltf_skin).
    usd.py         USD adapter (world transform, fan triangulation).
    mesh.py        OBJ/STL/OFF adapter via trimesh.

  stages/
    reconstruct.py Poisson reconstruction + subprocess isolation.
    filter.py      Proximity filter, density filter, thin-shell extrusion.
    flatness.py    PCA flat-cell detection, oriented box generation.
    decompose.py   CoACD wrapper, LOD tiers, walkable hulls, dedup.
    repair.py      Seam repair at octree cell boundaries.
    segment.py     Bone segmentation for rigged assets.
    splat.py       Covariance normals, inflation, octree partition, spatial extract.

  verify/
    raycast.py     Shared Moller-Trumbore ray-triangle intersection.
    probe.py       Raycast coverage probe.
    sweep.py       Capsule traversability sweep.
    seam.py        Seam snag detection (used by stages/repair.py).

  exporters/
    phys.py        .phys binary packing + int16 quantization.
    json.py        JSON debug companion.
    usd.py         USD Physics output.
    bundle.py      Full artifact bundle (phys + build-plan + analysis + resolved-config).

src/chitin_service/
  models.py        JobConfig. poisson_depth defaults to None (auto-resolved).
  app.py           FastAPI endpoint. poisson_depth passed through, not hardcoded.
  store.py         SQLite persistence. poisson_depth nullable, auto on None.
  worker.py        Job execution wrapper. Writes report.json, not full bundle.
  cli.py           Service CLI.
```

Public API surface unchanged: `extract`, `extract_from_arrays`, `extract_from_mesh`, `extract_from_rigged_mesh`, `Config`, `ExtractionResult`, `BuildPlan`, `read_phys`, `validate_phys`.

Service `poisson_depth` defaults to `None` (auto-resolved). The service worker writes `report.json` per job but doesn't yet produce the full artifact bundle — wiring `export_bundle` into the worker is a follow-up.

## Pre-refactor state (for reference)

## Phased plan

### Phase 0: analyze + resolve (the spine)

This is the load-bearing move. Everything else hangs off it.

**Create `analyze.py`** -- a pure-function module that examines input and produces facts.

```python
@dataclass(frozen=True)
class InputAnalysis:
    format: str                    # "ply", "obj", "glb", "usd", ...
    has_opacity: bool
    has_covariance: bool
    is_environment_likely: bool
    is_skinned: bool
    is_manifold: bool | None       # None if not cheaply determinable
    point_count: int
    face_count: int | None
    opacity_is_logit: bool
    bbox_volume: float
    inner_density_ratio: float     # for environment heuristic
```

`analyze_input(path) -> InputAnalysis` reads just enough of the file to produce this. No config mutation, no side effects. The current `_is_environment_scan()`, PLY attribute sniffing from `_extract_from_ply`, and skin detection from `_extract_from_skinned_or_static` all fold into this.

**Create `resolve.py`** -- turns user config + analysis into a frozen resolved config.

```python
@dataclass(frozen=True)
class ResolvedConfig:
    # Everything from Config, plus:
    poisson_depth: int             # Always resolved, never None
    thin_shell: bool               # Might have been auto-enabled
    surface_proximity_filter: float # Might have been auto-set to 5.0
    use_spatial_split: bool        # Decided from point_count
    use_seam_repair: bool          # Decided from use_spatial_split + config
    pipeline_path: str             # "splat", "mesh", "rigged", "usd"

    # Provenance: why each decision was made
    decisions: dict[str, str]      # e.g. {"thin_shell": "auto: environment detected"}
```

`resolve_config(config: Config, analysis: InputAnalysis) -> ResolvedConfig` is the single place where policy lives. The `decisions` dict makes every auto-override explainable. No downstream code mutates config after this point.

**The service stops drifting.** `chitin_service/worker.py` calls `resolve_config()` with its `JobConfig` mapped to a `Config`. The three hardcoded `poisson_depth=8` values become impossible because `ResolvedConfig.poisson_depth` is always computed, never defaulted.

**What changes in core.py:** The `extract_from_arrays()` function currently does:
- `_is_environment_scan()` -> mutate config (lines 572-584)
- Inline PLY attribute detection (scattered through `_extract_from_ply`)
- Implicit spatial split decision (line 614: `if len(positions) > config.spatial_split_threshold`)

All of that moves into `analyze_input()` + `resolve_config()`. The pipeline functions receive a `ResolvedConfig` and trust it.

**BuildPlan gets `decisions`.** Instead of `plan.detected["auto_thin_shell"] = True`, the resolved config's `decisions` dict carries the full explanation. BuildPlan tracks execution (stages, timing), not policy.

### Phase 1: explicit pipeline stages

With the spine in place, make each stage a callable with a typed signature.

```
src/chitin/
  stages/
    __init__.py
    ingest.py        # load + normalize to common representation
    reconstruct.py   # Poisson reconstruction, subprocess isolation
    filter.py        # proximity filter, density filter, thin-shell
    flatness.py      # PCA flat-cell detection, oriented box generation
    decompose.py     # CoACD wrapper, LOD tier generation
    repair.py        # seam repair (uses verification internally)
    segment.py       # bone segmentation for rigged assets
    quantize.py      # int16 quantization, AABB computation
```

Each stage takes typed input + `ResolvedConfig`, returns typed output, and appends to the `BuildPlan`. No stage reads or mutates global state.

The pipeline runner becomes straightforward:

```python
def compile(path: Path, config: Config) -> CompileResult:
    analysis = analyze_input(path)
    resolved = resolve_config(config, analysis)
    plan = BuildPlan(analysis=analysis, resolved=resolved)

    raw = ingest(path, analysis, plan)
    surface = reconstruct(raw, resolved, plan)
    filtered = filter_surface(surface, resolved, plan)
    hulls = decompose(filtered, resolved, plan)
    hulls = repair(hulls, raw, resolved, plan)
    hulls = deduplicate(hulls, plan)
    quantized = quantize(hulls, plan)

    return CompileResult(hulls=quantized, plan=plan, resolved=resolved)
```

The rigged path and spatial-split path are branches inside `reconstruct` and `decompose`, not separate top-level flows. The runner is always the same shape.

**core.py shrinks to the runner + the public API functions.** The 1,457-line file becomes ~200 lines of orchestration. Each stage module is 100-200 lines and independently testable.

### Phase 2: verification module

Unify `probe.py` and `sweep.py` under a shared foundation.

```
src/chitin/
  verify/
    __init__.py
    raycast.py       # shared Moller-Trumbore, AABB pre-filter
    probe.py          # coverage probe (moved from top-level)
    sweep.py          # traversability sweep (moved from top-level)
    seam.py           # seam snag detection (extracted from sweep.py)
```

`raycast.py` provides `ray_closest_hit()` and `ray_hits_any()`. Both probe and sweep currently implement their own ray-triangle intersection with per-hull AABB pre-filtering. The logic is near-identical -- consolidating it means one place to optimize (SIMD, BVH, etc.) and one place to test.

Seam repair in `stages/repair.py` imports from `verify/seam.py` for snag detection, making the "verification feeds repair" relationship explicit.

### Phase 3: input adapters

Extract the three inline adapters from core.py into a clean interface.

```
src/chitin/
  adapters/
    __init__.py       # AdapterResult type, registry
    ply.py            # from _extract_from_ply
    gltf.py           # from _extract_from_skinned_or_static + gltf_skin.py
    usd.py            # from _extract_from_usd
    mesh.py           # OBJ, STL, OFF via trimesh
    arrays.py         # from extract_from_arrays (numpy direct input)
```

Each adapter returns a common `AdapterResult`:

```python
@dataclass
class AdapterResult:
    positions: np.ndarray           # (N, 3) float64
    faces: np.ndarray | None        # (M, 3) int32, None for point clouds
    normals: np.ndarray | None
    scales: np.ndarray | None       # gaussian covariance
    rots: np.ndarray | None         # gaussian covariance
    opacity: np.ndarray | None
    skin: SkinData | None           # joint indices, weights, bone names, IBMs
```

After adaptation, the pipeline doesn't know or care whether the source was PLY, GLB, or USD. The `analyze_input()` function from Phase 0 reads enough to pick the adapter; the adapter does the full load.

### Phase 4: exporters

Move export logic out of `result.py` into standalone modules.

```
src/chitin/
  exporters/
    __init__.py
    phys.py           # from ExtractionResult.to_phys
    json.py           # from ExtractionResult.to_json
    usd.py            # from ExtractionResult.to_usd
```

`ExtractionResult.to_phys()` currently contains 100+ lines of binary packing. `to_usd()` imports pxr inline. Moving these out keeps the result type clean (data only) and makes each exporter independently testable.

The public API methods (`result.to_phys(path)`) remain as thin wrappers that delegate to the exporter module, so the API surface doesn't break.

### Phase 5: artifact bundle

Every compile optionally produces a full artifact set:

```
output/
  scene.phys           # the contract
  build-plan.json      # stages, timing, decisions
  analysis.json        # input facts
  resolved-config.json # final config with provenance
  probe.json           # coverage results (if --auto-verify)
  sweep.json           # traversability results (if --auto-verify)
```

The CLI gets `--bundle` / `-b` to produce the directory instead of a single file. The service writes requested artifacts plus `report.json` per job; it does not yet produce the full bundle (build-plan.json, analysis.json, resolved-config.json as separate artifacts). Wiring `export_bundle` into the service worker is a follow-up.

`build-plan.json` is the key artifact. It answers "why did Chitin do what it did?" with machine-readable evidence. This is the property that makes Chitin trustworthy in CI pipelines: you can diff build plans across runs and catch regressions in the compiler's own decisions, not just in the output geometry.

## Sequencing

| Phase | Depends on | Risk | Scope |
|-------|-----------|------|-------|
| 0: analyze + resolve | nothing | low (additive, no API break) | ~400 lines new, ~200 lines removed from core.py |
| 1: pipeline stages | Phase 0 | medium (moves code, touches all paths) | ~800 lines moved, core.py shrinks to ~200 |
| 2: verification module | Phase 1 (for repair integration) | low (moves existing code) | ~300 lines moved + shared raycast |
| 3: input adapters | Phase 0 (for AdapterResult type) | medium (touches every input path) | ~400 lines moved from core.py |
| 4: exporters | Phase 1 (for clean result type) | low (moves existing code) | ~300 lines moved from result.py |
| 5: artifact bundle | Phases 0-2 | low (additive) | ~150 lines new |

Phase 0 is the one that changes the architecture. Phases 1-4 are code movement that the new architecture makes obvious. Phase 5 is product surface.

The public API (`extract`, `extract_from_arrays`, `extract_from_mesh`, `Config`, `read_phys`) does not change at any phase. Existing callers keep working.

## What this does NOT include

- No new features. No ground plane detection, no new input formats, no new export targets.
- No async pipeline. The synchronous model is correct for local/CI use. Async is a deployment concern for the service.
- No plugin registration system. Adapters and exporters are internal modules, not user-extensible plugins. If that changes later, the module boundaries are already right.
- No changes to `.phys` format or runtime consumers. The contract stays stable.

## Success criteria

After this refactor:

1. `poisson_depth=8` is impossible to hardcode -- the service must go through `resolve_config()`.
2. `core.py` is under 300 lines.
3. Every auto-detection decision appears in `resolved-config.json` with a human-readable reason.
4. Each pipeline stage is independently testable with fixture data.
5. `probe` and `sweep` share a single ray-triangle intersection implementation.
6. A CI job can diff `build-plan.json` across commits and catch compiler behavior changes.
