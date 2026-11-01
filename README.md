# chitin

[![CI](https://github.com/Autarkis/chitin/actions/workflows/ci.yml/badge.svg)](https://github.com/Autarkis/chitin/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Chitin compiles meshes, point clouds, and Gaussian fields into convex collision
hulls. Use the hulls directly or save them as `.phys` sidecars for Python,
TypeScript, C#, and C++ readers.

- **Meshes:** OBJ, STL, GLB, and USD input; experimental skinned GLB support.
- **Scans and splats:** surface reconstruction, opacity filtering, and spatial
  partitioning in the Python compiler.
- **Browser compilation:** GLB and decoded Gaussian fields through CoACD and
  Poisson WebAssembly modules running in Workers.
- **Runtime integration:** Rapier and Three.js bindings, a Unity importer, and
  an Unreal hull-data importer.
- **Inspection:** CLI checks, compilation reports, collision LODs, and a browser
  demo for inspecting and simulating the result.

## Browser backends

The browser compiler uses WebAssembly. WebGPU decomposition is in development.
See the [architecture](docs/architecture.md) and
[numerical input contract](docs/policy-0.3-input-contract.md) for details.

## Install

```bash
pip install chitin              # mesh extraction (OBJ, GLB, STL)
pip install chitin[splat]       # + point cloud / gaussian splat extraction
pip install chitin[usd]         # + USD Physics output
pip install chitin[service]     # + local build service
pip install chitin[all]         # everything
```

The base install handles mesh inputs with just trimesh + CoACD. The `[splat]` extra adds Open3D for Poisson surface reconstruction from point clouds and gaussian splats.

Mesh compilation supports Python 3.12+. The `[splat]` extra currently requires
Python 3.12 because Open3D has no Python 3.13 wheel.

### Browser path

**Release status (2026-10-31):** the examples below describe the current source
API. The published lite/web packages are at 0.1.2; chitin-wasm is not yet on
npm. Use source builds and locally served native binaries for these examples.

The browser API compiles static meshes and decoded Gaussian fields:

```bash
npm install @autarkis/chitin-lite
```

CoACD and Poisson run as WebAssembly in a Worker. The output uses the same
`.phys` format as the Python compiler. The separate `@autarkis/chitin-wasm`
source package contains the native binaries.

**Mesh compilation** — `compileGlb()` accepts a `File`, `Blob`, buffer, or URL:

```typescript
import { ChitinCompiler } from "@autarkis/chitin-lite";
import coacdModuleUrl from "@autarkis/chitin-wasm?url";
import coacdWasmUrl from "@autarkis/chitin-wasm/coacd.wasm?url";

const compiler = new ChitinCompiler({
  wasm: { js: coacdModuleUrl, wasm: coacdWasmUrl },
  maxWorkers: 2,
});
const { phys, hulls, report } = await compiler.compileGlb(file, {
  componentPolicy: { maxHulls: 128 },
});
```

**Gaussian splat compilation** — `compileGaussianField()` takes splat
parameters (centers, scales, quaternions, opacities) and runs the full
pipeline: surface reconstruction via Poisson WASM, then CoACD decomposition:

```typescript
import poissonModuleUrl from "@autarkis/chitin-wasm/poisson?url";
import poissonWasmUrl from "@autarkis/chitin-wasm/poisson.wasm?url";

const compiler = new ChitinCompiler({
  wasm: {
    js: coacdModuleUrl, wasm: coacdWasmUrl,
    poissonJs: poissonModuleUrl, poissonWasm: poissonWasmUrl,
  },
});
const { phys, hulls } = await compiler.compileGaussianField(
  { centers, scales, quaternions, opacities },
  { poissonDepth: 7 },
);
```

See [`integrations/wasm-lite/`](integrations/wasm-lite/) for the full API
and [`examples/quickstart/`](examples/quickstart/) for the drag-and-drop Collider
Lab.

Artifact-fit sampling and the interactive component policy are documented in
[the interactive compiler budget](docs/usage.md#interactive-compiler-budget).

Use `chitin check <file>` to see which path a given input needs:

```
$ chitin check model.glb
file:       model.glb
format:     glb
vertices:   12,847
faces:      25,102
manifold:   yes
path:       either
  server:   pip install chitin
  browser:  npm install @autarkis/chitin-lite
reason:     manifold mesh, eligible for browser-side decomposition
```

Plain point-cloud/PLY file ingestion and non-manifold repair require the Python
pipeline today. Browser callers can compile already-decoded Gaussian fields via
`compileGaussianField()`, but the browser build cannot repair open meshes because
it intentionally omits that dependency.

## CLI

```bash
# extract colliders from a splat point cloud
chitin extract scene.ply -o scene.phys --opacity-threshold 0.5

# extract from a mesh
chitin extract model.obj -o colliders.phys --concavity 0.05

# environment scan (room, cave, outdoor scene)
chitin extract room.ply -o room.phys --density-quantile 0.3 --proximity-filter 5.0 --thin-shell

# multi-LOD: generate tiers at different concavity thresholds
chitin extract model.obj -o colliders.phys --concavity 0.05 --lod-concavities 0.1,0.3,0.5

# inspect a .phys file (shows LOD tiers if present)
chitin inspect colliders.phys

# validate binary integrity
chitin validate colliders.phys
```

## Library

```python
from chitin import extract, Config

config = Config(concavity=0.05, opacity_threshold=0.5)
result = extract("scene.ply", config)

result.to_phys("colliders.phys")  # primary output
result.to_json("colliders.json")  # debug companion
result.to_usd("colliders.usda")  # USD Physics (Isaac Sim, Omniverse)

# multi-LOD output (v3 .phys with tiered collision hulls)
config = Config(concavity=0.05, lod_concavities=[0.1, 0.3, 0.5])
result = extract("model.obj", config)
result.to_phys("colliders.phys")  # LOD 0 at 0.05, then tiers at 0.1, 0.3, 0.5
```

### From numpy arrays

```python
import numpy as np
from chitin import extract_from_arrays, Config

positions = np.random.randn(10000, 3).astype(np.float32)
result = extract_from_arrays(positions, config=Config())
```

### Read .phys back

```python
from chitin import read_phys, validate_phys

phys = read_phys("colliders.phys")
for hull in phys.hulls:  # LOD 0 (highest detail)
    print(hull.vertices.shape, hull.indices.shape)

# LOD tiers (if present)
for tier in phys.lod_tiers:
    print(f"concavity={tier.concavity}: {tier.hull_count} hulls")

# pick the tier closest to a target concavity
coarse = phys.lod_tier(0.3)

issues = validate_phys("colliders.phys")
```

## .phys format

The `.phys` binary format stores quantized convex hulls, optional per-bone
bind transforms, and collision LOD tiers. See [docs/phys.md](docs/phys.md).

### Collision LOD

Generate LOD tiers at different concavity thresholds, then select a tier for
the viewing distance or simulation budget. LOD 0 is always the highest-detail decomposition. Additional tiers are coarser and cheaper. v2 readers open a v3 file and get LOD 0 without changes.

Runtime tier *selection* by nearest concavity is available in every reader: `phys.lod_tier(...)` (Python), `selectLodHulls(phys, ...)` / `createColliders(..., { lodConcavity })` (TypeScript), `PhysAsset.SelectLod(...)` (Unity), and `UChitinPhysAsset::SelectLod(...)` (Unreal). Readers with no target pick default to LOD 0.

| Format | Extension | Use |
|--------|-----------|-----|
| Binary sidecar | `.phys` | Web, native engines, Rapier, custom loaders |
| JSON | `.json` | Debug companion, lightweight consumers |
| USD Physics | `.usda` | Isaac Sim, Omniverse, Kit-based tools |

## Engine integrations

All integrations read the same `.phys` binary with identical dequantization.

| Engine | Package | Path |
|--------|---------|------|
| Web (Three.js + Rapier) | [`@autarkis/chitin-web`](integrations/web/) | `integrations/web/` |
| Unity | `com.chitin.physics` | `integrations/unity/` |
| Unreal Engine | ChitinImporter plugin | `integrations/unreal/` |

### Web: Three.js + Rapier

```typescript
import RAPIER from "@dimforge/rapier3d-compat";
import { parsePhys } from "@autarkis/chitin-web";          // format only, no deps
import { addToWorld } from "@autarkis/chitin-web/rapier";  // Rapier bindings

const buffer = await fetch("/assets/scene.phys").then((r) => r.arrayBuffer());
const phys = parsePhys(buffer);
addToWorld(RAPIER, world, phys);
```

The package root (`@autarkis/chitin-web`) is dependency-free — just the `.phys`
parser and validator. The Rapier collider bindings live at
`@autarkis/chitin-web/rapier` and the Three.js debug meshes at
`@autarkis/chitin-web/three`, so a format-only consumer never pulls in Three or
Rapier.

A full working example with capsule walk controller and Playwright tests lives in [`integrations/walktest/`](integrations/walktest/). See [docs/usage.md](docs/usage.md#web-quickstart-ply-to-walkable-browser-scene) for the end-to-end walkthrough.

### Unity: drag-and-drop import

Drop a `.phys` file into your Assets folder. The `ScriptedImporter` creates a GameObject hierarchy with convex MeshColliders -- no code required. For rigged assets, hulls are grouped under bone-named parents.

Install via Package Manager: "Add package from disk" -> `integrations/unity/package.json`. See [docs/usage.md](docs/usage.md#unity-quickstart-drag-and-drop-phys-import) for runtime loading code.

### Unreal: asset import

The ChitinImporter plugin imports a `.phys` into a `UChitinPhysAsset` (dequantized hulls + bind poses) exposed to Blueprints. It does not yet build `UBodySetup` collision for you -- read the hull vertices/indices from the asset and create the collision bodies (`FKConvexElem` / `UBodySetup`) in C++ or Blueprint.

### Other engines

Use `parsePhys(buffer)` (TypeScript) or `PhysReader.Read(data)` (C#/C++) directly and pass each hull's vertices/indices to the engine's convex-collider API.

## Local build service

A single-machine build server with content-addressed caching. Jobs run synchronously in the current process -- suitable for local/CI use, not production.

```bash
pip install chitin[service]
chitin-server serve --port 8400
chitin-server submit model.glb
chitin-server download <job_id> -o ./output
```

## How it works

1. Loads input (PLY, OBJ, STL, GLB, USD, or raw arrays)
2. Filters by opacity for gaussian splat point clouds
3. Derives oriented normals from gaussian covariance (scale + rotation) when available, falls back to KD-tree estimation
4. Auto-detects environment scans (hollow-shell point distributions) and enables thin-shell + proximity filter. Use `--no-auto-environment` to disable
5. Partitions large scenes into octree cells (threshold: 50K points) with ghost-zone overlap for boundary continuity
6. Optionally inflates gaussian centers into disk samples along their two largest axes for better surface coverage
7. Reconstructs surface mesh via Poisson reconstruction (Open3D), with auto-selected depth per cell and subprocess crash isolation
8. For environment scans: proximity-filters closure surfaces and optionally extrudes a thin shell to prevent interior volume fill
9. PCA-based flatness detection replaces near-flat octree cells with oriented boxes instead of running CoACD
10. Decomposes remaining cells into convex hulls (CoACD), in a subprocess and pinned to one thread so the same input yields the same hulls; `--fast` unpins it for 2-4x on concave assets and gives up reproducibility
11. Seam repair: detects height discontinuities at octree cell boundaries, merges affected cells, and re-extracts for continuous coverage
12. Deduplicates cross-cell hulls by AABB IOU
13. If `lod_concavities` is set, runs additional decompositions at each threshold to produce LOD tiers
14. For rigged GLTF assets (experimental): reads joint weights directly from GLB binary, segments by dominant bone, generates per-bone hulls in bone-local space

## Limitations

- **Environment scan auto-detection can misfire.** Auto-enabling `--thin-shell` and `--proximity-filter` and the preflight scan hint are two separate checks over different point sets, and only the first can change behavior: it fires when the scene has a hollow interior, or enough wall coverage plus floor coverage, and stays off below a minimum point count or bounding-box volume. The preflight hint checks a sampled shell fraction and only prints a suggestion — seeing it does not mean the flags were applied. Use `--no-auto-environment` to disable the automatic path, or pass `--thin-shell --proximity-filter` explicitly regardless of what either check decides.
- **Rigged GLTF support is experimental.** Skinning is read directly from GLB binary (trimesh drops these attributes). Currently supports single-primitive meshes. Interleaved `byteStride` is handled, but multiple primitives and vertex reordering may produce incorrect bone segmentation.
- **Flat surfaces can over-decompose.** A PCA-based flatness detector (`--flatness-threshold`, default 0.9) replaces near-flat octree cells with oriented boxes instead of running CoACD. On the one measured scene (Mip-NeRF 360 Garden, 585 hulls) it did not fire on any cell. See `examples/utility-proof/README.md` for the run and the `--flatness-threshold 0` escape hatch.
- **No sparse voxel collision output yet.** Chitin currently emits convex hull artifacts, not viewer-native SVO/voxel grids for walk-mode raycasts.
- **Point-cloud dependencies require Python 3.12.** The mesh-only install also supports Python 3.13+.
- **FBX needs Blender.** trimesh has no FBX loader, so `chitin extract model.fbx` auto-converts the file to GLB via headless Blender (Blender must be on PATH) and extracts from that. `chitin convert` runs the same step explicitly. Without Blender, FBX extraction raises a clear error.
- **No physics material metadata.** Input formats (USD, GLTF) may carry material properties (friction, density, restitution) that chitin does not propagate to the output. Consumers must assign material properties manually.
- **Rigged runtime placement is bind-pose only.** The web `addToWorld()` helper bakes each rigged hull to its bone's bind (rest) pose on a single fixed body, and the Three.js debug meshes do the same. For *animated* skinning — hulls that track moving bones — use `createColliders()` and attach each `boneMap` entry to its own body, driven by your bone poses.

## License

MIT
