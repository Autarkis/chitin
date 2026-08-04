# Changelog

All notable changes to Chitin are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The `.phys` binary
format is versioned independently and noted where it changes.

## [Unreleased]

- **Fixed:** PLY meshes were silently routed through the point-cloud/Poisson
  path instead of the mesh path. `adapters.ply.load_ply` never populated
  `AdapterResult.faces`, and `analyze._analyze_ply` hardcoded
  `face_count=None`, so any `.ply` with a face element — the format
  `docs/usage.md` lists as a mesh format — took the same branch as an
  unstructured point cloud. The reader now parses the face element (ascii and
  binary little-/big-endian, n-gons fan-triangulated, `vertex_indices` and
  `vertex_index` both accepted), and PLY meshes take the mesh path. On the
  Stanford Bunny (35,947 vertices, 69,451 faces): coverage 0.9386 → 1.0, hulls
  105 → 20, wall clock 126.1s → 41.0s. Gaussian-splat and point-cloud PLY input
  (no face element) is unaffected.
- **Fixed:** decomposition was not reproducible. CoACD's search is
  OpenMP-parallel and its thread scheduling picks which decomposition it
  settles on, so the same mesh and config returned a different hull count and
  different bytes on every run (47/48/50 hulls, four distinct hashes, over four
  runs of one 3.6k-face mesh) — which made the manifest hashes, the output
  cache, and "same input bytes = same output bytes" untrue for every concave
  asset. The CoACD worker now runs single-threaded by default, the same fix
  Poisson already had. It costs 2-4x wall time on concave assets and nothing on
  near-convex ones. `--fast` / `Config(coacd_deterministic=False)` restores the
  old behaviour; the mode is recorded in `resolved-config.json` and the build
  plan, and `--profile robotics` rejects a build that used it.
- **Fixed:** the CoACD time budget was 15s, below the real decomposition time
  of ordinary concave inputs, so a build that should have produced 46 hulls
  silently shipped one bounding box instead (or, under `robotics`, failed).
  On the 192 MB gaussian-splat scene in `examples/utility-proof`, 34 of 39
  cells hit that budget: the collider was 34 bounding boxes and 28 hulls, where
  a budget that lets the decomposition finish yields 266. The budget is a stall
  backstop, not a quality knob: it now defaults to 300s and is settable per
  build with `--coacd-timeout` / `Config(coacd_timeout=…)`.
- **Fixed:** the service's cache key ignored the build profile, so a `robotics`
  request could be served an `interactive` build — coarser geometry, plus a
  copied verdict its strict acceptance policy had never evaluated. The profile
  is part of the key now, and an unknown profile is rejected with a `400` at
  submission instead of failing later in the worker.
- **Fixed:** rigged assets were rejected by every strict profile. The rigged
  path never recorded coverage, so `covered_fraction` was `None` and the
  coverage check could not pass. Coverage is now measured bind-posed against the
  model-space input, which also counts bones that were too small to decompose.
- **Fixed:** CoACD timeouts were invisible outside the single-mesh path. Per-bone,
  per-octree-cell and seam-repair decompositions ran without a build plan, so
  `fallback_hulls` read zero and a `robotics` build shipped bounding boxes it
  should have rejected. Each sub-decomposition now carries its counters back to
  the asset's plan (octree cells return theirs across the worker boundary).
- **Fixed:** a bundle's `manifest.json` listed whatever was in the output
  directory, hashing stale files from earlier runs, and missed `probe.json`
  because `--auto-verify` wrote it after the manifest. `export_bundle()` now
  tracks exactly what it wrote and takes a `post_export` hook so the probe lands
  first.
- **Fixed:** a profile preset overwrote an explicitly passed flag whose value
  happened to equal the default (`--concavity 0.05 --profile robotics`). The CLI
  now determines which flags were actually typed; `apply_profile()` takes an
  optional `explicit` set.
- **Fixed:** web `addToWorld()` created the rigid body before validating hulls,
  leaving a body with partial colliders in the world when a bone-local hull had
  no bind pose to place it.
- **Fixed:** `@autarkis/chitin-lite`'s worker client was permanently wedged by a
  synchronous `postMessage` failure (detached transfer buffer, uncloneable
  config) — the pending slot and abort listener were never released.
- **Fixed:** the manifold precheck only rejected triangles that repeated a
  vertex index; distinct but collinear or coincident points passed. It now
  applies a scale-relative area floor.
- Service reports carry `effective_config` alongside the requested `config`, so
  a profile-adjusted build no longer reports settings it wasn't built with.
- CI: all pull-request checks (tests, WASM build, dependency review, PR
  conventions) live in one workflow behind a single `gate` job, since branch
  protection can only aggregate jobs within a workflow. `pr.yml` and
  `build-wasm.yml` are folded into `ci.yml`.
- CI: every GitHub Action bumped to its current major — `checkout` v4→v7,
  `setup-node` v4→v7, `setup-python` v5→v7, `upload-artifact` v4→v7,
  `download-artifact` v4→v8, `dependency-review-action` v4→v5,
  `upload-pages-artifact` v3→v5, `deploy-pages` v4→v5, `configure-pages` v5→v6.
  `setup-emsdk` moves to its new home at `emscripten-core/setup-emsdk`.
- Target normalization now rescales gaussian-splat covariance. `target_height` /
  `target_footprint` previously scaled positions but left the per-splat scales
  alone, so splat inflation offsets and octree ghost-zone radii stayed in the
  source scale while the geometry was metric — under-covering the surface at the
  Poisson step. The same uniform factor is now applied to the scales, additively
  in log space or multiplicatively when `splat_scale_is_log` is `false`;
  rotations are untouched. The build plan records `normalize_covariance_scale`
  (replacing the `normalize_covariance_unscaled` warning flag). (#17)
- Web Rapier adapter: `addToWorld()` now places rigged `.phys` assets at their
  bind (rest) pose instead of refusing them — each bone-local hull is baked
  through its bone's bind transform (`world = local @ bind_transform`, row-major)
  before becoming a collider, so scale/shear in the matrix stay exact. Exposed as
  `applyBindPose()`; assets with bone indices but no bind poses still error
  clearly. (#14)
- Build profiles (`--profile interactive | walkable | robotics`): each profile
  sets preset core-config defaults and an acceptance gate. `interactive`
  (default) is permissive; `walkable`/`robotics` enforce coverage, and `robotics`
  additionally rejects CoACD-timeout bounding-box fallbacks. A build that fails a
  strict gate is rejected with a verdict instead of silently ending `COMPLETE`;
  the same policy evaluation runs on the CLI and service paths. (#19)
- Provenance manifest: `--bundle` writes a `manifest.json` (alongside
  `build-plan.json`, `analysis.json`, `resolved-config.json`) recording the
  manifest/`.phys` versions, compiler and dependency versions, input/output
  content hashes, and quality warnings, with a `verify_bundle()` checker. (#20)

## [0.1.2] - 2026-07-19

- Replace the GPL-licensed `plyfile` dependency with a small built-in permissive
  PLY reader, so the whole dependency stack is permissive (MIT/BSD) and cleanly
  usable in commercial software.
- Add `keywords`, `homepage`, and `bugs` to both npm packages so they surface in
  registry search (npm-only; the Python package already carried these).

## [0.1.1] - 2026-07-19

- Add a README for `@autarkis/chitin-web`.
- npm packages publish via GitHub OIDC trusted publishers instead of a token.

## [0.1.0] - 2026-07-19

First public release. `.phys` format version 3.

### Compiler (Python)

- Extract convex-hull colliders from meshes, point clouds, gaussian splats, and
  USD, emitting the `.phys` binary sidecar (plus JSON and USD Physics outputs).
- Gaussian-splat pipeline: covariance-derived normals, opacity filtering, octree
  spatial decomposition, Poisson reconstruction, CoACD convex decomposition,
  flatness detection, seam repair, and cross-cell reconciliation.
- Multi-LOD output: tiered decompositions at multiple concavity thresholds in one
  `.phys`.
- Real-world scale normalization (`--target-height` / `--target-footprint`),
  applied consistently across all extract entry points.
- Experimental rigged-GLTF support: per-bone hulls in bone-local space.
- FBX inputs auto-convert to GLB via headless Blender.
- Robustness: CoACD runs in a bounded subprocess with a bounding-box fallback so
  non-watertight input can never hang the pipeline; the spatial pool uses `spawn`.
- Local build service (`chitin-server`) with content-addressed caching, keyed on
  input kind + dependency versions.

### Format & readers

- `.phys` v3: quantized convex hulls, per-bone bind transforms, and LOD tiers.
- Readers for Python, TypeScript (`@autarkis/chitin-web`), Unity, and Unreal,
  all with nearest-concavity LOD tier selection.
- Hardened validation across Python and the web parser: rejects unknown
  versions/flags, trailing bytes, out-of-range or non-contiguous hull/LOD offsets,
  out-of-range bone indices, and non-finite AABBs and bind transforms.
- Cross-runtime conformance suite: one golden corpus verified by the Python and
  TypeScript readers.

### Browser

- `@autarkis/chitin-lite`: CoACD compiled to WebAssembly with a TypeScript API
  producing the same v3 `.phys` as the Python compiler. Typed errors and writer
  input validation.
- `@autarkis/chitin-web` uses subpath exports so the format reader is
  dependency-free; Rapier and Three.js bindings live under `/rapier` and `/three`.

### Engines

- Unity `com.chitin.physics` (drag-drop ScriptedImporter), Unreal ChitinImporter
  (asset import), and a Three.js + Rapier web bridge.

[0.1.2]: https://github.com/Autarkis/chitin/releases/tag/web-v0.1.2
[0.1.1]: https://github.com/Autarkis/chitin/releases/tag/python-v0.1.1
[0.1.0]: https://github.com/Autarkis/chitin/releases/tag/python-v0.1.0
