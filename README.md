# chitin

Convex collision geometry from point clouds, meshes, and gaussian splats.

Point cloud in, physics-ready convex hulls out. Outputs USD Physics, flat JSON, or binary `.phys` sidecar. Any physics engine can eat them.

## Install

```bash
pip install chitin            # core (JSON + binary output)
pip install chitin[usd]       # + USD Physics output
pip install chitin[all]       # everything
```

## Usage

### CLI

```bash
chitin input.ply -o physics.usda
chitin input.ply -o hulls.json --format json
chitin input.ply -o scene.phys --format phys --concavity 0.05
```

### Library

```python
from chitin import extract, Config

config = Config(concavity=0.05, opacity_threshold=0.5)
result = extract("input.ply", config)

result.to_usd("physics.usda")
result.to_json("hulls.json")
result.to_phys("scene.phys")
```

### From numpy

```python
import numpy as np
from chitin import extract_from_arrays, Config

positions = np.random.randn(10000, 3).astype(np.float32)
result = extract_from_arrays(positions, config=Config())
```

## What it does

1. Loads point cloud or mesh input (PLY, OBJ, STL, USD, or raw arrays)
2. Filters by opacity/density to drop transparent regions
3. Reconstructs a surface mesh via Poisson reconstruction (Open3D)
4. Decomposes into convex hulls via CoACD
5. Emits physics-ready collision geometry

## Output formats

| Format | File | Consumer |
|--------|------|----------|
| USD Physics | `.usda` | Kit, Isaac Sim, Unity, Unreal, Blender |
| Flat JSON | `.json` | USD-free consumers, web, lightweight integrations |
| Binary sidecar | `.phys` | Web/native players (Rapier, custom engines) |

## License

MIT
