# Usage Guide

## Installation

> **Status:** not published yet -- the current PyPI `chitin` is an unrelated placeholder and the npm packages are unpublished. Until release, install from source: clone the repo and `pip install -e .` (add extras, e.g. `pip install -e ".[splat]"`).

```bash
pip install chitin              # mesh extraction (OBJ, GLB, STL)
pip install chitin[splat]       # + point cloud / gaussian splat extraction
pip install chitin[usd]         # + USD Physics output
pip install chitin[service]     # + local build service
pip install chitin[all]         # everything
```

The base install handles mesh inputs with just trimesh + CoACD. The `[splat]` extra adds Open3D (Poisson surface reconstruction) and scipy for point cloud and gaussian splat extraction.

Requires Python 3.12. The base install works on 3.13+; `chitin[splat]` requires open3d which does not yet have a 3.13 wheel.

## Which flags do I need?

Most inputs work with defaults. Use `chitin check <file>` to see what chitin detects about your input, then pick a recipe:

| Input type | Example command | Notes |
|------------|----------------|-------|
| Gaussian splat (PLY with covariance) | `chitin extract scene.ply -o scene.phys` | Defaults handle opacity filtering, covariance normals, and spatial partitioning. Add `--opacity-threshold 0.5` if you want stricter filtering. |
| Room / environment scan | `chitin extract room.ply -o room.phys` | Point-cloud/splat input gets `--proximity-filter 5.0` by default. Auto-detected environments also get `--thin-shell`. Use `--no-auto-environment` to disable environment detection, `--proximity-filter 0` to disable proximity filtering. |
| Clean mesh (OBJ, GLB, STL) | `chitin extract model.obj -o model.phys` | Just works. Adjust `--concavity` to trade hull count for fit (lower = tighter). |
| Large mesh (200K+ verts) | `chitin extract big.obj -o big.phys` | Decimates above the `Config.max_decompose_vertices` field (200K default) **when Open3D is available** (`chitin[splat]`). On a base install without Open3D, decimation is skipped with a logged warning and the full mesh is passed to CoACD. Set the threshold via the Python `Config`; there is no CLI flag. |
| Multi-LOD | `chitin extract model.obj -o model.phys --lod-concavities 0.1,0.3,0.5` | LOD 0 uses `--concavity`; each additional threshold must be coarser (greater) than `--concavity`. Output is v3 `.phys`. |
| Rigged character (GLB) | `chitin extract character.glb -o character.phys` | Experimental. Per-bone hulls in bone-local space. Single-primitive GLB only. |
| FBX (static or skinned) | `chitin extract model.fbx -o model.phys` | trimesh has no FBX loader, so extract auto-converts FBX to GLB via headless Blender (must be on PATH). `chitin convert` does the step explicitly. |
| USD scene | `chitin extract scene.usda -o colliders.usda` | Requires `pip install chitin[usd]`. |

If you're unsure, start with defaults and inspect the result with `chitin inspect output.phys` and `chitin probe output.phys`.

## CLI

### Extract

```bash
chitin extract <input> -o <output> [options]
```

Supported inputs: `.ply`, `.obj`, `.stl`, `.off`, `.glb`, `.gltf`, `.usd`, `.usda`, `.usdc`, `.fbx`. FBX auto-converts to GLB via headless Blender (Blender must be on PATH; see `chitin convert`).

Supported outputs: `.phys` (binary sidecar), `.json` (debug companion), `.usda` (USD Physics)

For when to reach for Chitin versus a voxel/splat-viewer collision pipeline, see [Compared with viewer collision](../README.md#compared-with-viewer-collision) in the top-level README.

Because `.phys` is a sidecar, the visual runtime does not need to be Chitin-aware. A splat viewer, Three.js scene, generated-world renderer, or custom engine can load the visual asset however it wants, then load `scene.phys` in the same coordinate space and attach those hulls to its physics world.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--profile` | interactive | Build profile: preset defaults + acceptance gate. `interactive` is permissive; `walkable`/`robotics` are strict (see [Acceptance profiles](#acceptance-profiles)). |
| `--concavity` | 0.05 | CoACD concavity threshold. Lower = tighter fit, more hulls. |
| `--opacity-threshold` | 0.1 | Minimum opacity to keep a point (splat inputs only). |
| `--poisson-depth` | auto | Poisson reconstruction depth (point cloud inputs only). Auto-selects per cell based on point count. |
| `--max-hulls` | 2048 | Max convex hulls per decomposition unit (per source-mesh component / octree cell / bone), not a global cap. |
| `--lod-concavities` | none | Comma-separated concavity thresholds for LOD tiers. |
| `--density-quantile` | 0.1 | Poisson density filter quantile. Raise to 0.3+ for environments. |
| `--proximity-filter` | auto (5.0 for point-cloud/splat input) | Remove mesh vertices farther than N * median_nn_distance from input. 0 disables. |
| `--thin-shell` | off | Extrude surface into thin solid before decomposition (environment scans). |
| `--thin-shell-thickness` | 0 | Shell thickness (0 = auto from mesh extent). |
| `--scene-name` | scene | Root prim name (USD output only). |
| `--force` | off | Run even if preflight check flags the input as too large. |
| `-q, --quiet` | off | Suppress progress output. |
| `--flatness-threshold` | 0.9 | PCA eigenvalue ratio to classify octree cells as flat (0 = disabled). |
| `--auto-verify` | off | Run raycast probe after extraction and print coverage summary. |
| `--no-auto-environment` | off | Disable auto-detection of environment scans. |
| `--no-seam-repair` | off | Disable seam repair pass at octree cell boundaries. |
| `--snug-fit` | off | Tighten hull face planes onto covered input points (experimental). |
| `--target-height` | none | Uniformly rescale the input so its height (up-axis extent) is N meters before extraction (for non-metric source assets). |
| `--target-footprint` | none | Real-world footprint (largest horizontal extent, meters) used instead of `--target-height` for flat objects like rugs. |
| `--up-axis` | 1 | Which axis (0/1/2) is up/height for `--target-height` (default 1, glTF Y-up). |
| `-b, --bundle` | off | Write full artifact bundle (scene.phys + build-plan.json + analysis.json + resolved-config.json + manifest.json) to a directory instead of a single file. |
| `--no-hook` | off | Skip post-process hook. |
| `--fast` | off | Let CoACD use every core. 2-4x faster on concave assets, but its search then varies run to run, so the build is not reproducible and `--profile robotics` rejects it. See [Reproducible output](#reproducible-output). |
| `--coacd-timeout` | 300 | Seconds before a single CoACD call is killed and a bounding box substituted. A backstop against a native stall, not a quality knob. |

**Examples:**

```bash
# gaussian splat point cloud -> binary colliders
chitin extract scene.ply -o scene.phys --opacity-threshold 0.5

# mesh -> colliders with tight concavity
chitin extract model.obj -o colliders.phys --concavity 0.01

# multi-LOD: one file with 4 detail tiers
chitin extract model.obj -o colliders.phys \
    --concavity 0.05 \
    --lod-concavities 0.1,0.3,0.5

# USD Physics output for Isaac Sim / Omniverse
chitin extract scan.ply -o colliders.usda

# full artifact bundle (phys + build plan + analysis + resolved config + provenance manifest)
chitin extract model.obj -o out.phys --bundle
# writes model_bundle/ with scene.phys, build-plan.json, analysis.json, resolved-config.json, manifest.json

# strict robotics profile: tighter decomposition + acceptance gate (exit 3 if rejected)
chitin extract arm.glb -o arm.phys --profile robotics --bundle
```

### Acceptance profiles

`--profile` selects preset build defaults *and* an acceptance policy that
decides whether the result is good enough to ship. The CLI records the verdict
(pass/fail with reasons) in the bundle's `manifest.json` under `quality.verdict`
and prints the failing reasons to stderr; the service also writes it to the
job's `report.json`.

| Profile | Presets | Accepts unless… |
|---------|---------|-----------------|
| `interactive` (default) | none | never rejects (permissive) |
| `walkable` | coarser concavity (0.1), denser Poisson filtering | coverage below 85% |
| `robotics` | tight concavity (0.01), snug fit | any CoACD-timeout bounding-box fallback, no hulls, coverage below 90%, or the build ran `--fast` (not reproducible) |

A profile only fills fields you did not set, so an explicit `--concavity`
always overrides the profile's preset -- including when the value you pass is
the same as the default, which the CLI tells apart by checking which flags you
actually typed. A strict profile that rejects a build exits non-zero (`3`) and
skips the post-process hook.

### Reproducible output

Chitin's promise is that the same input bytes and the same config give the same
output bytes — that is what makes the manifest hashes, the output cache, and
a re-run of an old build meaningful.

One dependency does not cooperate by default. CoACD's search is multithreaded,
and its thread scheduling decides which decomposition it settles on: the same
mesh yields a different hull count and different bytes on every run (measured on
one 3.6k-face mesh: 47, 48, 50, 47 hulls over four runs, four distinct hashes).
Chitin therefore runs CoACD pinned to a single thread. Reproducibility costs
2-4x wall time on a single concave mesh; on convex or near-convex ones it costs
nothing, because the search never branches. A scene pays much less, because its
cells decompose in parallel and the pool absorbs the difference: a 39-cell
gaussian-splat scene went from 180s to 286s, and its `.phys` came out
byte-identical across runs where the unpinned build produced a different hash
every time.

`--fast` gives the time back and takes the guarantee away. Use it for a preview
or a throwaway build, not for anything whose hashes you intend to trust: the
mode is recorded in `resolved-config.json` and in the build plan, and
`--profile robotics` rejects a build that used it.

```bash
chitin extract chair.glb -o chair.phys                 # reproducible (default)
chitin extract chair.glb -o chair.phys --fast          # faster, not reproducible
chitin extract chair.glb -o chair.phys --profile robotics --fast   # rejected
```

Recover the throughput at the batch level instead — several assets compiling in
parallel, each pinned — rather than by unpinning one asset.

### Provenance manifest

Every bundle carries a `manifest.json` tying the build together: the input
SHA-256, a SHA-256 for each emitted file, the compiler version and shaping-
dependency versions (CoACD, trimesh, numpy, Open3D), the effective config (its
values and their hash, after any profile presets were applied), the
auto-resolved config, the `.phys` format version, and the acceptance verdict.
Only files this build wrote are listed, and the manifest is written last, so it
covers every one of them — including `probe.json` when `--auto-verify` is on.
It makes a bundle tamper-evident — recompute the declared hashes to confirm the
artifacts match what the manifest claims:

```python
from chitin import verify_bundle

problems = verify_bundle("model_bundle")  # [] means every artifact matches
```

The manifest also carries the versioned cross-runtime compilation report under
`quality.report`. Its typed metrics, warnings, verdict state, runtime identity,
and reproducibility scope are documented in
[`compilation-report.md`](compilation-report.md). A consumer must distinguish
`verdict.status: "not_evaluated"` from a profile pass.

### Inspect

```bash
chitin inspect <file.phys>
```

Prints format version, hull count, vertex/triangle totals, bone info, and per-hull dimensions. For v3 files with LOD tiers, prints a table per tier.

**Sample output (multi-LOD):**

```
version:    3
hulls:      235
vertices:   21305
triangles:  42596
rigged:     False
lod_tiers:  3

LOD 0: 235 hulls
  hull 0: 128 verts, 252 tris, size [0.045, 0.038, 0.041]
  hull 1: 96 verts, 188 tris, size [0.032, 0.029, 0.035]
  ...

LOD 1 (concavity=0.100): 80 hulls
  hull 0: 92 verts, 180 tris, size [0.078, 0.065, 0.071]
  ...

LOD 2 (concavity=0.300): 20 hulls
  ...

LOD 3 (concavity=0.500): 8 hulls
  ...
```

### Check

```bash
chitin check <input>
```

Reports the input format, vertex/face counts, and which processing path is needed (server Python pipeline vs. browser WASM). For PLY files, detects opacity and covariance attributes. For meshes, checks manifold status.

### Validate

```bash
chitin validate <file.phys>
```

Checks structural integrity: magic bytes, version, offset consistency, index bounds, AABB sanity, bind-pose block completeness, LOD block data sizes. Exits with code 1 if any errors are found.

### Probe

```bash
chitin probe <file.phys> [--grid 64] [--capsule-radius 0.3] [-o results.json]
```

Raycast coverage probe. Fires a grid of downward rays through the scene AABB and reports what percentage hit collision geometry. Classifies gaps by capsule radius. Exits with code 2 on low confidence.

### Sweep

```bash
chitin sweep <file.phys> [--grid 32] [--capsule-radius 0.3] [--capsule-height 1.8] [--step-height 0.3]
```

Capsule traversability test. Resolves each grid column to a ground surface, drops the cells the capsule cannot occupy, builds an adjacency graph filtered by step height, flood-fills connected components, and reports what fraction of standable ground is reachable from the largest island. Rates results as excellent (>=95%), good (>=80%), fair (>=50%), or poor (<50%). Exits with code 2 on poor rating.

Both capsule dimensions are enforced:

- `--capsule-height` picks the ground. Every hull a column crosses contributes a solid span; overlapping spans merge, and the ground is the lowest span top with at least this much free space above it. A floor roofed lower than the capsule is skipped in favour of the next surface up, so cells under a table resolve to the table top rather than the floor beneath it. Those cells are counted as `clearance_blocked`.
- `--capsule-radius` clears the sides. Eight samples on a ring of this radius reject the cell when geometry crosses the band between step height and head height above its ground -- a capsule jammed against a wall or a table edge is not standable. Those cells are counted as `radius_blocked`. The radius also trims the outer grid margin and deduplicates snag points.

`ground_cells` still counts every column that hit geometry at all; `standable_cells` is the subset the capsule fits in, and is the denominator of the reported fraction.

One caveat: free space above the topmost surface is unbounded, so a room whose ceiling is lower than the capsule resolves to the roof rather than reporting no floor. The `clearance_blocked` count makes that case visible -- when it approaches `standable_cells`, the sweep is walking a roof, not a floor.

### Convert

```bash
chitin convert <input.fbx> [-o output.glb]
```

Converts FBX to GLB via Blender headless (requires Blender on PATH). Useful as a preprocessing step for skinned FBX files before extraction.

## Python API

### Basic extraction

```python
from chitin import extract, Config

result = extract("scene.ply", Config(concavity=0.05, opacity_threshold=0.5))

result.to_phys("colliders.phys")
result.to_json("colliders.json")
result.to_usd("colliders.usda")

print(f"{len(result.hulls)} hulls from {result.source_vertex_count} source verts")
```

### Multi-LOD extraction

```python
from chitin import extract, Config

config = Config(
    concavity=0.05,
    lod_concavities=[0.1, 0.3, 0.5],
)
result = extract("model.obj", config)
result.to_phys("colliders.phys")  # v3 file with 4 tiers total
```

LOD 0 uses the primary `concavity` value. Each entry in `lod_concavities` produces an additional tier at that threshold. The output `.phys` file is v3 with the `HAS_LOD` flag.

### From numpy arrays

```python
import numpy as np
from chitin import extract_from_arrays, Config

positions = np.random.randn(10000, 3).astype(np.float64)
opacity = np.random.rand(10000).astype(np.float64)

result = extract_from_arrays(
    positions,
    opacity=opacity,
    config=Config(opacity_threshold=0.3),
)
```

### From an existing mesh

```python
import numpy as np
from chitin import extract_from_mesh, Config

vertices = np.array([...], dtype=np.float32)  # (N, 3)
faces = np.array([...], dtype=np.int32)        # (M, 3)

result = extract_from_mesh(vertices, faces, config=Config(concavity=0.1))
```

### Reading .phys files

```python
from chitin import read_phys, validate_phys

phys = read_phys("colliders.phys")

# LOD 0 hulls (highest detail, always present)
for hull in phys.hulls:
    print(hull.vertices.shape, hull.indices.shape)
    print(f"  aabb: {hull.aabb_min} -> {hull.aabb_max}")

# LOD tiers (empty list if v2 or no LOD)
for tier in phys.lod_tiers:
    print(f"concavity={tier.concavity}: {tier.hull_count} hulls, {tier.total_vertices} verts")

# pick the tier closest to a target concavity
coarse = phys.lod_tier(0.3)
if coarse:
    for hull in coarse.hulls:
        # use hull.vertices, hull.indices
        pass

# bone info (rigged assets)
if phys.bones:
    for bone in phys.bones:
        print(f"{bone.name}: {bone.bind_transform.shape}")

# validation
issues = validate_phys("colliders.phys")
for issue in issues:
    print(issue)  # "[error] hull 3: index 412 >= vertex_count 400"
```

### Browser runtime

`@autarkis/chitin-web` reads `.phys` files and turns them into runtime objects for browser scenes. Use `addToWorld` for the common Rapier path, or `parsePhys` directly if your viewer uses another physics engine. The package uses subpath exports: the root is dependency-free (`parsePhys`, `selectLodHulls`), the Rapier bindings live at `@autarkis/chitin-web/rapier`, and the Three.js debug meshes at `@autarkis/chitin-web/three`.

`@autarkis/chitin-lite` also compiles a self-contained GLB directly in the
browser without blocking the main thread:

```typescript
import { ChitinCompiler } from "@autarkis/chitin-lite";

const compiler = new ChitinCompiler({
  wasm: { js: "/coacd/coacd.mjs", wasm: "/coacd/coacd.wasm", version: "0.2.0" },
  maxWorkers: 2,
});
const { phys, hulls, source, report } = await compiler.compileGlb(file, {
  profile: "interactive",
  componentPolicy: { maxHulls: 128 },
  onProgress: ({ stage, completed, total, eta_ms }) =>
    console.log(stage, completed, total, eta_ms),
});
compiler.terminate();
```

Inputs may be `File`, `Blob`, `ArrayBuffer`, an array-buffer view, `URL`, or a
URL string. The compiler reads the active scene, applies node transforms and
instancing, and supports indexed, unindexed, interleaved, and sparse static
triangle geometry. It welds exact render seams and decomposes disconnected
triangle components independently before writing one sidecar. Unsupported
shape-changing features fail explicitly. The
browser API does not yet claim a profile pass: `report.verdict.status` is
`not_evaluated` until the corresponding artifact checks exist.

For benchmark and acceptance lanes, pass
`quality: { surfaceSamples: 2048, volumeSamples: 8192 }`. The compiler then
records deterministic sampled source-surface coverage, worst connected-part
coverage, collider-volume precision, raw false-fill fraction, and
clearance-aware deep false fill in
`report.metrics`. Sampling is disabled by default so interactive recompilation
does not pay the verification cost. Measurements alone do not change the
verdict from `not_evaluated`.

#### Interactive compiler budget

The interactive compiler keeps disconnected parts isolated because assembled
GLBs can contain overlapping solids that CoACD cannot safely process as one
mesh. It nevertheless treats the hull count as a scene budget: large parts are
processed first, while a part below both the default 20% scene-diagonal and
0.5% scene-volume cutoffs receives one convex approximation. The default total
budget has a 128-hull fine-detail ceiling. Between thresholds `0.10` and `0.60`,
the effective budget scales down to 70% of the capacity remaining above
deterministic per-component minimums. Explicit caller budgets are not rescaled.
Its bounded search uses 8 MCTS nodes, 40 iterations, and depth 2;
callers can override those through `decompose`. When 128 hulls cannot satisfy
the per-component minimums, the default expands rather than dropping
geometry. Any small-part simplification is recorded as
`INTERACTIVE_SMALL_COMPONENTS_SIMPLIFIED`. These decisions depend only on
geometry and configuration, not on wall-clock time, and are recorded in
`report.config.effective` and `warnings`.

The interactive planner also identifies closed components whose enclosed
volume is at most 5% of their AABB volume. These low-occupancy shells commonly
represent plates, bowls, covers, and other hollow forms where an aggressively
coarse convex decomposition can bridge an opening or fill a cavity. By default
they retain a threshold no coarser than `0.05` and reserve at least eight hull
slots. The total hull budget can still decrease at coarser detail settings, so
the control changes collider complexity without removing the cavity guard.
`INTERACTIVE_HOLLOW_SHELL_GUARD` explains when a requested slider value was
limited. This is a deterministic planning heuristic; because the browser
profile does not yet measure free-space clearance, it does not claim that the
result passed a cavity-preservation check.

For other detailed components below 50% enclosed-volume/AABB occupancy, the
planner also scales the coarsest permitted threshold by scene importance. A
component that dominates the scene receives a default ceiling near `0.14`;
progressively smaller non-trivial components may use up to twice that value.
This prevents a high slider setting from collapsing a primary body with
attached features into one convex envelope while leaving ordinary compact
bodies unconstrained. `INTERACTIVE_IMPORTANCE_GUARD` records when this limit
applies. `componentPolicy.importantComponentMaxThreshold` configures the
scene-dominant baseline and `importantComponentMaxOccupancyRatio` configures
eligibility.

The planner separately controls vertices inside each hull. Scene-relative
diagonal is the primary allocation, refined by the component's isoperimetric
quotient as a bounded geometric roundness/curvature proxy. This keeps large
curved geometry smooth while making small flat parts such as fins materially
cheaper. Defaults range from 8 to 96 vertices per hull.
`decompose.maxChVertex` acts as an additional global ceiling, while
`componentPolicy.minHullVertices` and `maxHullVertices` tune the adaptive range.
`INTERACTIVE_HULL_VERTICES_ADAPTED` records when this changes component caps.
In the browser compiler, this vertex adaptation requires a CoACD WASM build
with convex-hull decimation enabled.

Reusing the same `ChitinCompiler` and immutable `File`/`Blob` also reuses parsed
geometry and compatible completed components across detail changes. Set
`componentPolicy.enabled` to `false` for uniform full-detail-per-part behavior,
or tune `maxHulls`, `smallComponentMaxDiagonalRatio`,
`smallComponentMaxVolumeRatio`, `smallComponentThreshold`, and
`detailedComponentMinThreshold`, `importantComponentMaxThreshold`,
`importantComponentMaxOccupancyRatio`, `hollowShellMaxOccupancyRatio`,
`hollowShellMaxThreshold`, `hollowShellMinHulls`, `minHullVertices`, and
`maxHullVertices` explicitly. Remaining hull capacity is assigned using both
scene importance and normalized source complexity so a detailed articulated
shell is not starved by a simpler sibling with a larger AABB. The default
minimum is `0.10`; set
it to `0` when an application deliberately accepts unbounded fine-detail waits.
`decompose.maxConvexHull` remains a per-component low-level ceiling;
`componentPolicy.maxHulls` is the only total scene budget.
The Collider Lab keeps the last completed collider visible while a replacement
detail setting compiles, then reveals the replacement only after it is ready.

```typescript
import RAPIER from "@dimforge/rapier3d-compat";
import { parsePhys } from "@autarkis/chitin-web"; // format only, no deps
import { addToWorld } from "@autarkis/chitin-web/rapier";
import { createDebugMeshes } from "@autarkis/chitin-web/three";

const buffer = await fetch("/assets/scene.phys").then((r) => r.arrayBuffer());
const phys = parsePhys(buffer);

addToWorld(RAPIER, world, phys);
scene.add(createDebugMeshes(phys));
```

### Web quickstart: PLY to walkable browser scene

End-to-end from a gaussian splat scan to collision working in Three.js + Rapier:

**Step 1: Generate collision**

```bash
pip install chitin[splat]    # requires Python 3.12
chitin extract scene.ply -o scene.phys
chitin inspect scene.phys    # verify hull count looks reasonable
```

**Step 2: Load in your Three.js scene**

```typescript
import RAPIER from "@dimforge/rapier3d-compat";
import { parsePhys } from "@autarkis/chitin-web"; // format only, no deps
import { addToWorld } from "@autarkis/chitin-web/rapier";
import { createDebugMeshes } from "@autarkis/chitin-web/three";

// after RAPIER.init() and world creation:
const buffer = await fetch("/assets/scene.phys").then((r) => r.arrayBuffer());
const phys = parsePhys(buffer);

addToWorld(RAPIER, world, phys);          // fixed convex colliders
scene.add(createDebugMeshes(phys));       // green wireframe overlay for debugging
```

The visual splat loads however your viewer handles it. The `.phys` sidecar just needs to share the same coordinate space -- no coupling between the two loaders.

A complete working example (Three.js + Rapier + capsule walk controller + Playwright tests) lives in [`integrations/walktest/`](../integrations/walktest/). Build and run it with:

```bash
cd integrations/walktest
npm install && npm run build
npx serve harness    # open http://localhost:3000
```

Then call `__walktest.loadPhys("/path/to/scene.phys")` from the browser console.

### Unity quickstart: drag-and-drop .phys import

The `com.chitin.physics` package includes a `ScriptedImporter` that auto-imports `.phys` files as GameObjects with convex MeshColliders.

**Step 1: Install the UPM package**

In Unity's Package Manager, choose "Add package from disk" and select `integrations/unity/package.json`. Or add to your `Packages/manifest.json`:

```json
{
  "dependencies": {
    "com.chitin.physics": "file:../../integrations/unity"
  }
}
```

**Step 2: Import**

Drag a `.phys` file into your Unity project's Assets folder. The importer creates:
- A root GameObject (`<name>_colliders`)
- One child per hull, each with a convex `MeshCollider`
- For rigged assets: child objects grouped under bone-named parents with bind transforms applied

No code required for the basic case. For runtime loading:

```csharp
using Chitin;

byte[] data = File.ReadAllBytes("scene.phys");
PhysAsset phys = PhysReader.Read(data);

foreach (PhysHull hull in phys.hulls)
{
    var mesh = new Mesh();
    mesh.SetVertices(hull.vertices);
    mesh.SetTriangles(hull.triangles, 0);
    mesh.RecalculateNormals();

    var go = new GameObject($"hull");
    var mc = go.AddComponent<MeshCollider>();
    mc.sharedMesh = mesh;
    mc.convex = true;
}
```

### World-space reconstruction (rigged)

Hulls from rigged assets are in bone-local space. To get world coordinates:

```python
import numpy as np
from chitin import read_phys

phys = read_phys("character.phys")
for hull in phys.hulls:
    if hull.bone_index is not None:
        bone = phys.bones[hull.bone_index]
        local = hull.vertices
        ones = np.ones((len(local), 1), dtype=np.float32)
        world = (np.hstack([local, ones]) @ bone.bind_transform)[:, :3]
```

## Config Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `concavity` | float | 0.05 | CoACD concavity threshold for LOD 0. Lower = more hulls, tighter fit. |
| `opacity_threshold` | float | 0.5 | Minimum opacity to keep a point (splat inputs). |
| `poisson_depth` | int or None | None | Poisson reconstruction depth (point cloud inputs). None = auto-select per cell based on point count. Manual override 4-7 recommended; depths of 8+ are accepted but run in an isolated subprocess, since Open3D can segfault nondeterministically at high depth. |
| `min_hull_vertices` | int | 4 | Discard hulls with fewer vertices than this. |
| `max_hulls` | int | 2048 | Max convex hulls per decomposition unit (per source-mesh component / octree cell / bone), not a global cap. |
| `opacity_is_logit` | bool | False | Set True if opacity values are logits (pre-sigmoid). Auto-detected for PLY inputs. |
| `coacd_preprocess_mode` | str | "auto" | CoACD preprocessing mode. |
| `coacd_preprocess_resolution` | int | 50 | CoACD preprocessing resolution. |
| `coacd_deterministic` | bool | True | Pin CoACD to one thread so the same input gives the same hulls and the same bytes. False (`--fast`) is 2-4x quicker on concave assets and not reproducible. See [Reproducible output](#reproducible-output). |
| `coacd_timeout` | float | 300.0 | Seconds before a single CoACD call is killed and an axis-aligned bounding box substituted for it. Guards against a native stall; set it below the real decomposition time and output silently coarsens. |
| `max_decompose_vertices` | int | 200000 | Decimate mesh before decomposition if it exceeds this count. |
| `lod_concavities` | list[float] or None | None | Additional concavity thresholds for LOD tiers. Produces a v3 .phys file. |
| `splat_scale_is_log` | bool | True | Whether splat scale values are log-space (standard 3DGS convention). |
| `splat_surface_ratio` | float | 0.2 | Anisotropic inflation ratio for splat disk samples. Set to 0 to disable inflation. |
| `spatial_split_threshold` | int | 50000 | Point count above which octree spatial decomposition is used. |
| `poisson_density_quantile` | float | 0.1 | Poisson density filter quantile. Raise to 0.3+ for environment scans to strip closure surfaces. |
| `surface_proximity_filter` | float \| None | None | Max distance (as multiple of median NN distance) from input points. Removes Poisson closure geometry far from real data. Unset auto-resolves to 5.0 for point-cloud/splat reconstructions (0.0 for mesh/GLB input); explicit 0.0 disables. |
| `thin_shell` | bool | False | Extrude filtered surface into a thin watertight solid before CoACD. Prevents volume-fill on environment scans. |
| `thin_shell_thickness` | float | 0.0 | Shell extrusion thickness. 0 = auto (2% of median mesh extent). |
| `flatness_threshold` | float | 0.9 | PCA eigenvalue ratio to classify octree cells as flat. Flat cells get oriented boxes instead of CoACD. 0 = disabled. |
| `auto_environment` | bool | True | Auto-detect environment scans and enable thin-shell + proximity filter. Set False to disable. |
| `force_environment` | bool | False | Treat the input as an environment scan whatever detection says (`--environment`). |
| `seam_repair` | bool | True | Re-merge octree cells at seam boundaries to eliminate height discontinuities. |

## Gaussian Splat Covariance

When a PLY file contains `scale_0/1/2` and `rot_0/1/2/3` attributes (standard 3DGS output), chitin uses the covariance data in two ways:

1. **Oriented normals**: The shortest axis of each gaussian's scale ellipsoid points along the surface normal. This produces better normals than KD-tree estimation, because the trainer already learned the surface orientation.

2. **Anisotropic inflation**: Each gaussian center is expanded into disk samples along its two largest axes, scaled by `splat_surface_ratio`. This gives the Poisson reconstructor better surface coverage -- fewer holes, tighter hulls. The default ratio of 0.2 adds 4 samples per point (5x total).

For PLY files without covariance attributes and no `face` element (plain point clouds, photogrammetry), chitin falls back to the standard pipeline: KD-tree normal estimation and no inflation. A PLY with a `face` element is a mesh and skips reconstruction entirely, taking the same direct decompose path as OBJ/GLB input.

Covariance travels with the geometry under `target_height` / `target_footprint`: the same uniform factor that rescales the positions is applied to the per-splat scales, so inflation offsets and octree ghost-zone radii stay in the units of the normalized cloud. `splat_scale_is_log` (default `true`, the 3DGS convention) decides how — an additive `log(factor)` on log scales, a plain multiply on linear ones. Rotations are untouched; a uniform scale does not reorient a gaussian. The build plan records the applied factor as `normalize_covariance_scale`.

### Spatial Decomposition for Large Scenes

When a splat scene exceeds `spatial_split_threshold` points (default 50K), chitin automatically partitions the scene into octree cells and processes each cell independently. This keeps each cell's point count manageable for Poisson reconstruction, avoids hitting the `max_decompose_vertices` decimation limit, and enables natural parallelism.

Each cell is padded by a ghost zone (3x the 95th-percentile splat radius in that cell) so that boundary geometry is reconstructed in both adjacent cells. Because padding is computed per cell, cells with small splats get tight ghost zones while cells with larger splats get wider ones. After per-cell decomposition, a reconciliation pass deduplicates hulls at boundaries using AABB IOU (threshold 0.5), keeping the larger hull when two overlap significantly.

Poisson reconstruction depth is auto-selected per cell based on point count (`floor(log2(n) / 3)`, clamped to 4-7). This avoids over-resolution on small cells and under-resolution on dense ones. Each cell's Poisson step runs in a subprocess so that an Open3D segfault on one cell doesn't kill the entire pipeline -- the cell is skipped and remaining cells continue. A manual `poisson_depth` of 8 or higher is likewise forced into a subprocess even on the non-partitioned path, so a high-depth segfault can never take down the compiler process.

The build plan tracks `cell_count`, `padding_min`, `padding_median`, `padding_max`, and `reconciled_hulls` for diagnostics.

### Environment Scans

Poisson reconstruction produces watertight meshes. For object scans (a mug, a statue), this is correct -- the closed surface IS the collision boundary. For environment scans (a room, a cave, an outdoor scene), Poisson closes the open boundaries and CoACD decomposes the enclosed volume, filling walkable space with invisible collision blocks.

Chitin auto-detects environment scans on two signals, either of which is enough. The first is a hollow middle: fewer than 5% of points in the inner 50% of the scene AABB. The second is a shell signature -- a floor plane plus at least two wall planes found against the AABB faces, each one thin (the points hug a single depth rather than filling the slab, which is what separates a wall from a solid block) and covering at least 35% of its face. The second signal is what carries cluttered interiors: a central pillar, a heap of material, or a mid-floor row of shelving raises inner density past 5%, but the walls and floor are still there.

When either fires, thin-shell extrusion is enabled automatically (and proximity filtering, if not already on by the point-cloud default below, is set to 5.0). Use `--no-auto-environment` or `auto_environment=False` to disable, and `--environment` or `force_environment=True` to force the environment path when detection misses. Inputs that land between the two -- inner density in [0.05, 0.20) with no shell signature -- are treated as solid objects, and `chitin check` says so and points at `--environment`.

Proximity filtering is not gated on environment detection: any point-cloud or splat input that goes through Poisson reconstruction gets `surface_proximity_filter=5.0` by default, whether or not it looks like a room, since Poisson closure geometry is artificial regardless of scene shape. Mesh/GLB input never runs Poisson and is unaffected. Pass `--proximity-filter 0` (or `surface_proximity_filter=0.0`) to disable it explicitly on any input.

Two mechanisms address the closure problem:

**Proximity filtering** (`surface_proximity_filter`): removes reconstructed mesh vertices that are far from any actual input point. Poisson's closure surfaces are artificial geometry with no nearby source data, so a distance threshold strips them while preserving real surfaces.

**Thin-shell extrusion** (`thin_shell`): after filtering, extrudes the remaining surface into a thin watertight solid (inner + outer surface + stitched boundary edges). CoACD decomposes this thin slab instead of the full enclosed volume, producing collision hulls that follow the wall/floor/ceiling surfaces rather than filling the interior.

```bash
# auto-detection handles most cases -- just run extract
chitin extract room.ply -o room.phys

# explicit environment config (if auto-detection is off or needs tuning)
chitin extract room.ply -o room.phys \
    --density-quantile 0.3 \
    --proximity-filter 5.0 \
    --thin-shell

# disable auto-detection for a scene that looks hollow but isn't
chitin extract hollow-object.ply -o out.phys --no-auto-environment
```

```python
# auto-detection (default)
config = Config(concavity=0.05)

# explicit environment config
config = Config(
    concavity=0.05,
    auto_environment=False,
    poisson_density_quantile=0.3,
    surface_proximity_filter=5.0,
    thin_shell=True,
)
```

For object scans reconstructed from a point cloud, the proximity filter default (5.0) still applies, but thin shell stays off. Environment auto-detection is conservative: it only triggers thin-shell for clearly hollow distributions.

## Concavity Tuning

The `concavity` parameter controls how aggressively CoACD decomposes the mesh. Lower values chase surface detail more closely and produce more hulls; higher values approximate more coarsely with fewer hulls.

Measured hull counts, three real glTF-Sample-Assets meshes (fetched by `examples/quickstart/scripts/prepare-samples.mjs`), one `chitin extract --concavity <value>` run per cell, hull count read from `chitin inspect`:

| Concavity | barramundi-fish (2188 verts) | clearcoat-wicker (1728 verts) | iridescent-dish-with-olives (14863 verts) | Use case |
|-----------|:---:|:---:|:---:|----------|
| 0.01      | 32  | 1*  | 1*  | Precise simulation, close interaction |
| 0.05      | 6   | 1   | 47  | General purpose, good balance |
| 0.1       | 3   | 1   | 19  | Background objects, mobile |
| 0.3       | 1   | 1   | 5   | Broadphase, simple collision |
| 0.5       | 1   | 1   | 1   | Bounding approximation |

\* clearcoat-wicker and iridescent-dish-with-olives at concavity 0.01 both hit the 300s CoACD-timeout backstop (`--coacd-timeout`, default 300) and fell back to a single bounding-box hull (8 verts, 12 tris) rather than completing a real decomposition; the "1" is a timeout artifact, not a converged result.

Hull count depends on mesh complexity, not on concavity alone: at 0.05, the low-poly clearcoat-wicker mesh produces 1 hull while the higher-poly iridescent-dish-with-olives produces 47. These three small assets (57 KB-461 KB) are not a general benchmark — do not extrapolate these exact counts to arbitrary meshes; re-measure on your own asset if you need a hull budget.

Environment: chitin 0.1.2, CoACD via the `coacd` package (1.0.11), Windows 11, Python 3.12.11.

Direct mesh inputs are decomposed one vertex-connected component at a time.
That preserves author-provided solid boundaries in formats such as GLB and
prevents overlapping, individually watertight primitives from being handed to
CoACD as one invalid solid. `coacd_preprocess_mode="off"` therefore applies to
each eligible solid rather than to the combined triangle soup.

For multi-LOD, set `concavity` to your tightest tier and `lod_concavities` to progressively coarser values. The consumer picks the right tier at runtime based on distance, platform, or simulation budget.

## Post-Process Hooks

Chitin can run a shell command after extraction. Configure globally in `~/.config/chitin/config.toml`:

```toml
[hooks]
post_process = "my-tool process {input}"
```

Or per invocation:

```bash
chitin extract model.ply -o out.phys --post-process "my-tool process {input}"
```

`{input}` is replaced with the input file path. Use `--no-hook` to skip.
