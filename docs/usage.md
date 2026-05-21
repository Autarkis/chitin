# Usage Guide

## Installation

```bash
pip install chitin              # mesh extraction (OBJ, GLB, STL, FBX)
pip install chitin[splat]       # + point cloud / gaussian splat extraction
pip install chitin[usd]         # + USD Physics output
pip install chitin[service]     # + local build service
pip install chitin[all]         # everything
```

The base install handles mesh inputs with just trimesh + CoACD. The `[splat]` extra adds Open3D (Poisson surface reconstruction) and scipy for point cloud and gaussian splat extraction.

Requires Python 3.12. The base install works on 3.13+; `chitin[splat]` requires open3d which does not yet have a 3.13 wheel.

## CLI

### Extract

```bash
chitin extract <input> -o <output> [options]
```

Supported inputs: `.ply`, `.obj`, `.stl`, `.off`, `.glb`, `.gltf`, `.fbx`, `.usd`, `.usda`, `.usdc`

Supported outputs: `.phys` (binary sidecar), `.json` (debug companion), `.usda` (USD Physics)

Use Chitin when the collision result needs to be a portable physics asset: checked into a build, validated in CI, loaded by multiple runtimes, or inspected independently from the visual source. If the only goal is immediate walk mode inside a splat viewer, a voxel collision pipeline may be enough. Chitin is for the next step: reusable convex hull artifacts with stable readers.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--concavity` | 0.05 | CoACD concavity threshold. Lower = tighter fit, more hulls. |
| `--opacity-threshold` | 0.1 | Minimum opacity to keep a point (splat inputs only). |
| `--poisson-depth` | auto | Poisson reconstruction depth (point cloud inputs only). Auto-selects per cell based on point count. |
| `--max-hulls` | 2048 | Maximum number of convex hulls. |
| `--lod-concavities` | none | Comma-separated concavity thresholds for LOD tiers. |
| `--density-quantile` | 0.1 | Poisson density filter quantile. Raise to 0.3+ for environments. |
| `--proximity-filter` | 0 | Remove mesh vertices farther than N * median_nn_distance from input. |
| `--thin-shell` | off | Extrude surface into thin solid before decomposition (environment scans). |
| `--thin-shell-thickness` | 0 | Shell thickness (0 = auto from mesh extent). |
| `--scene-name` | scene | Root prim name (USD output only). |
| `--force` | off | Run even if preflight check flags the input as too large. |
| `-q, --quiet` | off | Suppress progress output. |
| `--no-hook` | off | Skip post-process hook. |

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
```

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

### Validate

```bash
chitin validate <file.phys>
```

Checks structural integrity: magic bytes, version, offset consistency, index bounds, AABB sanity, bind-pose block completeness, LOD block data sizes. Exits with code 1 if any errors are found.

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
| `poisson_depth` | int or None | None | Poisson reconstruction depth (point cloud inputs). None = auto-select per cell based on point count. Manual override: 4-7. |
| `min_hull_vertices` | int | 4 | Discard hulls with fewer vertices than this. |
| `max_hulls` | int | 2048 | Maximum number of convex hulls. |
| `opacity_is_logit` | bool | False | Set True if opacity values are logits (pre-sigmoid). Auto-detected for PLY inputs. |
| `coacd_preprocess_mode` | str | "auto" | CoACD preprocessing mode. |
| `coacd_preprocess_resolution` | int | 50 | CoACD preprocessing resolution. |
| `max_decompose_vertices` | int | 200000 | Decimate mesh before decomposition if it exceeds this count. |
| `lod_concavities` | list[float] or None | None | Additional concavity thresholds for LOD tiers. Produces a v3 .phys file. |
| `splat_scale_is_log` | bool | True | Whether splat scale values are log-space (standard 3DGS convention). |
| `splat_surface_ratio` | float | 0.2 | Anisotropic inflation ratio for splat disk samples. Set to 0 to disable inflation. |
| `spatial_split_threshold` | int | 50000 | Point count above which octree spatial decomposition is used. |
| `poisson_density_quantile` | float | 0.1 | Poisson density filter quantile. Raise to 0.3+ for environment scans to strip closure surfaces. |
| `surface_proximity_filter` | float | 0.0 | Max distance (as multiple of median NN distance) from input points. Removes Poisson closure geometry far from real data. 0 = disabled. |
| `thin_shell` | bool | False | Extrude filtered surface into a thin watertight solid before CoACD. Prevents volume-fill on environment scans. |
| `thin_shell_thickness` | float | 0.0 | Shell extrusion thickness. 0 = auto (2% of median mesh extent). |

## Gaussian Splat Covariance

When a PLY file contains `scale_0/1/2` and `rot_0/1/2/3` attributes (standard 3DGS output), chitin uses the covariance data in two ways:

1. **Oriented normals**: The shortest axis of each gaussian's scale ellipsoid points along the surface normal. This produces better normals than KD-tree estimation, because the trainer already learned the surface orientation.

2. **Anisotropic inflation**: Each gaussian center is expanded into disk samples along its two largest axes, scaled by `splat_surface_ratio`. This gives the Poisson reconstructor better surface coverage -- fewer holes, tighter hulls. The default ratio of 0.2 adds 4 samples per point (5x total), which is a good balance between coverage and computation.

For PLY files without covariance attributes (plain point clouds, photogrammetry), chitin falls back to the standard pipeline: KD-tree normal estimation and no inflation.

### Spatial Decomposition for Large Scenes

When a splat scene exceeds `spatial_split_threshold` points (default 50K), chitin automatically partitions the scene into octree cells and processes each cell independently. This keeps each cell's point count manageable for Poisson reconstruction, avoids hitting the `max_decompose_vertices` decimation limit, and enables natural parallelism.

Each cell is padded by a ghost zone (3x the 95th-percentile splat radius in that cell) so that boundary geometry is reconstructed in both adjacent cells. Because padding is computed per cell, cells with small splats get tight ghost zones while cells with larger splats get wider ones. After per-cell decomposition, a reconciliation pass deduplicates hulls at boundaries using AABB IOU (threshold 0.5), keeping the larger hull when two overlap significantly.

Poisson reconstruction depth is auto-selected per cell based on point count (`floor(log2(n) / 3)`, clamped to 4-7). This avoids over-resolution on small cells and under-resolution on dense ones. Each cell's Poisson step runs in a subprocess so that an Open3D segfault on one cell doesn't kill the entire pipeline -- the cell is skipped and remaining cells continue.

The build plan tracks `cell_count`, `padding_min`, `padding_median`, `padding_max`, and `reconciled_hulls` for diagnostics.

### Environment Scans

Poisson reconstruction produces watertight meshes. For object scans (a mug, a statue), this is correct -- the closed surface IS the collision boundary. For environment scans (a room, a cave, an outdoor scene), Poisson closes the open boundaries and CoACD decomposes the enclosed volume, filling walkable space with invisible collision blocks.

Two mechanisms address this:

**Proximity filtering** (`surface_proximity_filter`): removes reconstructed mesh vertices that are far from any actual input point. Poisson's closure surfaces are artificial geometry with no nearby source data, so a distance threshold strips them while preserving real surfaces.

**Thin-shell extrusion** (`thin_shell`): after filtering, extrudes the remaining surface into a thin watertight solid (inner + outer surface + stitched boundary edges). CoACD decomposes this thin slab instead of the full enclosed volume, producing collision hulls that follow the wall/floor/ceiling surfaces rather than filling the interior.

```bash
# environment scan with both fixes
chitin extract room.ply -o room.phys \
    --density-quantile 0.3 \
    --proximity-filter 5.0 \
    --thin-shell
```

```python
config = Config(
    concavity=0.05,
    poisson_density_quantile=0.3,
    surface_proximity_filter=5.0,
    thin_shell=True,
)
```

For object scans, the defaults (no proximity filter, no thin shell) remain correct.

## Concavity Tuning

The `concavity` parameter controls how aggressively CoACD decomposes the mesh. Think of it as a fidelity budget:

| Concavity | Hulls (typical) | Use case |
|-----------|----------------|----------|
| 0.01      | 500+           | Precise simulation, close interaction |
| 0.05      | 100-300        | General purpose, good balance |
| 0.1       | 50-100         | Background objects, mobile |
| 0.3       | 10-30          | Broadphase, simple collision |
| 0.5       | 5-10           | Bounding approximation |

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
