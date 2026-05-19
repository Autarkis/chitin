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

# inspect a .phys file
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
for hull in phys.hulls:
    print(hull.vertices.shape, hull.indices.shape)

issues = validate_phys("colliders.phys")
```

## .phys format

The `.phys` v2 binary sidecar is the primary output. It stores quantized convex hulls with optional per-bone bind transforms in a single file that loads in microseconds. Full spec in [docs/phys.md](docs/phys.md).

| Format | Extension | Use |
|--------|-----------|-----|
| Binary sidecar | `.phys` | Web, native engines, Rapier, custom loaders |
| JSON | `.json` | Debug companion, lightweight consumers |
| USD Physics | `.usda` | Isaac Sim, Omniverse, Kit-based tools |

## Engine integrations

All integrations read the same `.phys` v2 binary with identical dequantization.

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
5. For rigged GLTF assets (experimental): reads joint weights directly from GLB binary, segments by dominant bone, generates per-bone hulls in bone-local space

## Limitations

- **Rigged GLTF support is experimental.** Skinning is read directly from GLB binary (trimesh drops these attributes). Currently supports single-primitive meshes with tightly packed accessors. Interleaved `byteStride`, multiple primitives, and vertex reordering may produce incorrect bone segmentation.
- **Python 3.12 only** until open3d ships a 3.13 wheel.
- **FBX skinning is not supported.** Static FBX meshes work via trimesh; skinned FBX requires a different code path.

## License

MIT
