# chitin

Open-source physics asset compiler for scanned, generated, splat, and rigged 3D assets.

Chitin is a free MIT-licensed compiler that bridges the gap between visual capture (gaussian splats, photogrammetry, LiDAR) and physics simulation. Feed it a point cloud, mesh, or skinned model and get back portable convex hulls that any engine can load. The primary output is the `.phys` binary sidecar -- a compact format with readers for Python, TypeScript, C#, and C++.

It is not a splat viewer feature or a single-engine import button. Viewer collision tools are great for making one splat scene walkable; Chitin's job is to turn messy 3D assets into deterministic, validated physics artifacts that can ship through web, engine, simulation, and CI pipelines.

## Why Chitin

- **Free MIT infrastructure**: audit it, vendor it, modify it, or run it offline without service lock-in.
- **Portable artifact contract**: `.phys` stores quantized convex hulls, bind transforms, and collision LOD tiers instead of engine-owned caches or viewer-only collision data.
- **Attachable to any viewer**: load a `.phys` sidecar next to a splat, mesh, or generated scene and feed the hulls into your runtime physics API.
- **Broader input surface**: splats, point clouds, static meshes, USD assets, and experimental rigged GLB support.
- **Thin runtime readers**: Python, TypeScript, C#, and C++ consumers load the same binary format while the heavy reconstruction/decomposition work stays in the compiler.
- **Pipeline-friendly checks**: `chitin check`, `inspect`, and `validate` make collision generation scriptable and reviewable.

## Use cases

- **Gaussian splat scenes**: extract collision geometry from PLY point clouds with opacity filtering
- **Robotics simulation**: generate colliders for scanned environments (Isaac Sim, Gazebo, MuJoCo)
- **Web/XR**: load `.phys` sidecars in the browser alongside your 3D viewer (Three.js + Rapier)
- **Game engines**: import `.phys` directly in Unity or Unreal with included plugins
- **Rigged characters**: per-bone convex hulls in bone-local space, ready for ragdoll or hit detection

## Compared with viewer collision

Splat viewer pipelines usually voxelize a gaussian splat, fill or carve navigable space, and feed that occupancy data to a specific viewer/runtime. That is the right shape for immediate walk mode.

Chitin reconstructs surfaces and decomposes them into convex hulls. That is the right shape when the output needs to become a reusable physics asset: versioned, validated, loadable by multiple engines, and independent of the original viewer. A splat viewer can keep its visual format and load Chitin collision as a sidecar.

## Install

```bash
pip install chitin              # mesh extraction (OBJ, GLB, STL, FBX)
pip install chitin[splat]       # + point cloud / gaussian splat extraction
pip install chitin[usd]         # + USD Physics output
pip install chitin[service]     # + local build service
pip install chitin[all]         # everything
```

The base install handles mesh inputs with just trimesh + CoACD. The `[splat]` extra adds Open3D for Poisson surface reconstruction from point clouds and gaussian splats.

Requires Python 3.12. (`chitin[splat]` requires open3d, which does not yet have a 3.13 wheel. The base install works on 3.13+.)

### Browser path

For manifold meshes (OBJ, GLB, STL from standard modeling tools), you can skip Python entirely and run decomposition in the browser:

```bash
npm install @autarkis/chitin-lite
```

This wraps CoACD compiled to WebAssembly (558KB) and writes the same `.phys` format the Python compiler produces. See [`integrations/wasm-lite/`](integrations/wasm-lite/) for usage.

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

Point clouds, gaussian splats, and non-manifold meshes require the Python pipeline.

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

result.to_phys("colliders.phys")   # primary output
result.to_json("colliders.json")   # debug companion
result.to_usd("colliders.usda")    # USD Physics (Isaac Sim, Omniverse)

# multi-LOD output (v3 .phys with tiered collision hulls)
config = Config(concavity=0.05, lod_concavities=[0.1, 0.3, 0.5])
result = extract("model.obj", config)
result.to_phys("colliders.phys")   # LOD 0 at 0.05, then tiers at 0.1, 0.3, 0.5
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
for hull in phys.hulls:              # LOD 0 (highest detail)
    print(hull.vertices.shape, hull.indices.shape)

# LOD tiers (if present)
for tier in phys.lod_tiers:
    print(f"concavity={tier.concavity}: {tier.hull_count} hulls")

# pick the tier closest to a target concavity
coarse = phys.lod_tier(0.3)

issues = validate_phys("colliders.phys")
```

## .phys format

The `.phys` binary sidecar is the primary output. It stores quantized convex hulls with optional per-bone bind transforms and collision LOD tiers in a single file that loads in microseconds. Full spec in [docs/phys.md](docs/phys.md).

### Collision LOD

A single decomposition forces a tradeoff between fidelity and cost. Multi-LOD solves this: the producer generates tiers at different concavity thresholds, the consumer picks based on distance, platform budget, or simulation context. LOD 0 is always the highest-detail decomposition. Additional tiers are coarser and cheaper. v2 readers open a v3 file and get LOD 0 without changes.

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

### Web runtime snippet

Use `@autarkis/chitin-web` anywhere you already have a browser physics world. The visual asset can be a splat, GLB, generated mesh, or anything else; the `.phys` file just needs to share the same coordinate space.

```typescript
import RAPIER from "@dimforge/rapier3d";
import { parsePhys, addToWorld } from "@autarkis/chitin-web";

const buffer = await fetch("/assets/scene.phys").then((r) => r.arrayBuffer());
const phys = parsePhys(buffer);

// Adds fixed Rapier convex colliders at the visual scene origin.
addToWorld(RAPIER, world, phys);
```

For other physics runtimes, use `parsePhys(buffer)` directly and pass each hull's vertices/indices to the engine's convex-collider API.

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
10. Decomposes remaining cells into convex hulls (CoACD)
11. Seam repair: detects height discontinuities at octree cell boundaries, merges affected cells, and re-extracts for seamless coverage
12. Deduplicates cross-cell hulls by AABB IOU
13. If `lod_concavities` is set, runs additional decompositions at each threshold to produce LOD tiers
14. For rigged GLTF assets (experimental): reads joint weights directly from GLB binary, segments by dominant bone, generates per-bone hulls in bone-local space

## Limitations

- **Environment scan auto-detection can misfire.** Chitin auto-detects hollow-shell point distributions and enables `--thin-shell` and `--proximity-filter` automatically. Use `--no-auto-environment` to disable if the heuristic is wrong for your scene.
- **Rigged GLTF support is experimental.** Skinning is read directly from GLB binary (trimesh drops these attributes). Currently supports single-primitive meshes. Interleaved `byteStride` is handled, but multiple primitives and vertex reordering may produce incorrect bone segmentation.
- **Flat surfaces over-decompose (mitigated).** A PCA-based flatness detector (`--flatness-threshold`, default 0.9) replaces near-flat octree cells with oriented boxes instead of running CoACD. On the Mip-NeRF 360 Garden scene this reduced hulls from 1,725 to 579 and build time from 27 min to 9 min. Scenes with unusual ground geometry may need threshold tuning or `--flatness-threshold 0` to disable.
- **No sparse voxel collision output yet.** Chitin currently emits convex hull artifacts, not viewer-native SVO/voxel grids for walk-mode raycasts.
- **Python 3.12 only** until open3d ships a 3.13 wheel.
- **FBX skinning is not directly supported.** Static FBX meshes work via trimesh. For skinned FBX, use `chitin convert` to convert to GLB via Blender headless, then extract from the GLB.
- **No physics material metadata.** Input formats (USD, GLTF) may carry material properties (friction, density, restitution) that chitin does not propagate to the output. Consumers must assign material properties manually.

## License

MIT
