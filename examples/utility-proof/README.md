# Utility Proof

Empirical validation that Chitin produces useful physics artifacts from real-world 3D assets.

## Question

Can Chitin take messy real 3D input and produce physics artifacts that are useful enough to ship, inspect, and reuse?

## Test cases

| Case | Dataset | Tests |
|------|---------|-------|
| Scanned object | Stanford Bunny, YCB mug | Static mesh → convex hulls. Visual fit, hull count, sim stability |
| Gaussian splat scene | Mip-NeRF 360 Garden (3DGS) | Splat point cloud → scene-scale collision. Opacity filtering, covariance normals, octree partitioning, Poisson reconstruction + decomposition |
| Rigged character | Microsoft Rocketbox | Per-bone hulls, bind transform correctness, ragdoll hull placement |

## Run

```bash
# 1. download assets (reads datasets.toml)
python download.py                  # all datasets
python download.py scanned-object   # just one

# 2. run chitin against each asset
python run_proof.py                 # all downloaded
python run_proof.py ycb-mug         # just one

# 3. check results
python run_proof.py --list
cat reports/ycb-mug/report.json
cat reports/ycb-mug/inspect.txt
```

## Output per run

```
reports/<key>/
  report.json     — metrics: hull count, build time, file sizes, validation status
  colliders.phys  — generated .phys sidecar
  colliders.json  — JSON companion
  inspect.txt     — chitin inspect output
  validate.txt    — chitin validate output
```

## Results

### Mip-NeRF 360 Garden (3DGS)

773,074 gaussian splat vertices through the full pipeline (opacity filter, covariance normals, octree partition, Poisson per cell, density filter, IOU dedup, CoACD, quantize):

| Metric | Value |
|--------|-------|
| Source vertices | 773,074 |
| Octree cells | 31 |
| Raw hulls (pre-dedup) | 2,181 |
| Final hulls | 1,725 |
| Dedup removed | 456 (21%) |
| Output size | 2.4 MB |
| Total vertices | 138,555 |
| Total triangles | 270,210 |
| Runtime | 27 min (M1 Pro, 16 cores) |
| Validation | CLEAN |

**Terrain explosion confirmed.** Cell 18 (ground plane, 115K triangles) produced 285 hulls alone -- 13% of the total budget from a single flat surface. At least 4 cells produced 100+ hulls from near-flat geometry, consuming ~40% of the hull budget on surfaces that could be represented by a single planar box each.

The IOU dedup removed 21% of raw hulls at cell boundaries, working as designed. The remaining problem is not boundary duplication but over-decomposition of flat surfaces.

## What success looks like

- Build completes without errors (subprocess isolation means individual cell crashes are tolerated, not fatal)
- `chitin validate` passes clean
- Hull count is reasonable for the asset complexity (not 1, not 10000)
- Build time is under 60s for typical objects. Large splat scenes (500K+ points, dozens of octree cells) take 15-30 minutes -- the bottleneck is per-cell Poisson reconstruction and CoACD decomposition, not chitin overhead
- .phys file size is small relative to the source asset

Failures are expected and useful. A bad result becomes the roadmap: better defaults, better diagnostics, better preflight, or a documented limitation.

## Datasets

Assets are NOT checked into git. `datasets.toml` is the manifest; `download.py` fetches them. Each entry records URL, license, citation, format, and redistribution status.

## Notes

Requires Python 3.12 with chitin installed (`pip install -e .` from repo root).
