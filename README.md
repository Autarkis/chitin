# chitin

Convex collision geometry from point clouds, meshes, and gaussian splats.

Asset in, physics-ready convex hulls out. Primary output is the `.phys` binary sidecar. Also emits JSON (debug companion) and USD Physics (Omniverse/Isaac Sim). Supports static meshes, point clouds with opacity, and skinned/rigged assets with per-bone colliders.

## Install

```bash
pip install chitin              # core (phys + JSON output)
pip install chitin[usd]         # + USD Physics output
pip install chitin[service]     # + build service (FastAPI)
pip install chitin[all]         # everything
```

## CLI

```bash
# extract colliders
chitin extract input.ply -o physics.phys
chitin extract model.glb -o hulls.json --format json
chitin extract scan.ply -o scene.usda --concavity 0.05

# inspect a .phys file
chitin inspect colliders.phys

# validate binary integrity
chitin validate colliders.phys
```

## Library

```python
from chitin import extract, Config

config = Config(concavity=0.05, opacity_threshold=0.5)
result = extract("input.ply", config)

result.to_phys("colliders.phys")   # primary output
result.to_json("colliders.json")   # debug companion
result.to_usd("colliders.usda")    # USD Physics
```

### From numpy

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

## Build service

A local build service with job state machine, content-addressed caching, and artifact storage.

```bash
pip install chitin[service]

# start server
chitin-server serve --port 8400

# submit a job (defaults to phys,json output)
chitin-server submit model.glb --wait

# check status / download artifacts
chitin-server status <job_id>
chitin-server download <job_id> -o ./output
```

## What it does

1. Loads input (PLY, OBJ, STL, GLB, GLTF, FBX, USD, or raw arrays)
2. Detects skinning (GLTF joint weights read directly from binary)
3. Filters by opacity for gaussian splat point clouds
4. Reconstructs surface mesh via Poisson reconstruction (Open3D)
5. Decomposes into convex hulls via CoACD
6. For rigged assets: segments by dominant bone, generates per-bone hulls in bone-local space
7. Emits physics-ready collision geometry with build report

## Output formats

| Format | Extension | Primary use |
|--------|-----------|-------------|
| Binary sidecar | `.phys` | Web, native engines, Rapier, custom loaders |
| JSON | `.json` | Debug companion, web, lightweight consumers |
| USD Physics | `.usda` | Isaac Sim, Omniverse, Kit-based tools |

The `.phys` format is documented in [docs/phys.md](docs/phys.md).

## Engine integrations

| Engine | Package | Path |
|--------|---------|------|
| Web (Three.js + Rapier) | `@chitin/web` | `integrations/web/` |
| Unity | `com.chitin.physics` | `integrations/unity/` |
| Unreal Engine | ChitinImporter plugin | `integrations/unreal/` |

All three read the same `.phys` v2 binary format with identical dequantization.

## License

MIT
