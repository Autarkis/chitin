# chitin

Physics-ready colliders from scanned, generated, and rigged 3D assets.

Chitin bridges the gap between visual capture (gaussian splats, photogrammetry, LiDAR) and physics simulation. Feed it a point cloud, mesh, or skinned model and get back portable convex hulls that any engine can load. The primary output is the `.phys` binary sidecar -- a compact format with readers for Python, TypeScript, C#, and C++.

## Use cases

- **Gaussian splat scenes**: extract collision geometry from PLY point clouds with opacity filtering
- **Robotics simulation**: generate colliders for scanned environments (Isaac Sim, Gazebo, MuJoCo)
- **Web/XR**: load `.phys` sidecars in the browser alongside your 3D viewer (Three.js + Rapier)
- **Game engines**: import `.phys` directly in Unity or Unreal with included plugins
- **Rigged characters**: per-bone convex hulls in bone-local space, ready for ragdoll or hit detection

## Install

```bash
pip install chitin              # core (phys + JSON output)
pip install chitin[usd]         # + USD Physics output
pip install chitin[service]     # + local build service
pip install chitin[all]         # everything
```

Requires Python 3.12. (open3d does not yet have a 3.13 wheel.)

## CLI

```bash
# extract colliders from a splat point cloud
chitin extract scene.ply -o scene.phys --opacity-threshold 0.5

# extract from a mesh
chitin extract model.obj -o colliders.phys --concavity 0.05

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
3. Reconstructs surface mesh via Poisson reconstruction (Open3D)
4. Decomposes into convex hulls (CoACD)
5. If `lod_concavities` is set, runs additional decompositions at each threshold to produce LOD tiers
6. For rigged GLTF assets (experimental): reads joint weights directly from GLB binary, segments by dominant bone, generates per-bone hulls in bone-local space

## Limitations

- **Rigged GLTF support is experimental.** Skinning is read directly from GLB binary (trimesh drops these attributes). Currently supports single-primitive meshes with tightly packed accessors. Interleaved `byteStride`, multiple primitives, and vertex reordering may produce incorrect bone segmentation.
- **Python 3.12 only** until open3d ships a 3.13 wheel.
- **FBX skinning is not supported.** Static FBX meshes work via trimesh; skinned FBX requires a different code path.

## License

MIT
