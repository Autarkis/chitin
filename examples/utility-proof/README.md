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

### Flatness detection (Garden, `--flatness-threshold` disabled vs default)

Same asset, same `--concavity 0.05`, run back to back on one machine. This is a `.ply` splat/point-cloud input, so it takes the octree/splat path; Leg A passes `--flatness-threshold 0` (disables the PCA flatness path per `src/chitin/stages/splat.py:208`, `if config.flatness_threshold > 0` — the mesh path gates the same way at `src/chitin/stages/repair.py:85`); leg B uses the default `0.9`.

| Metric | Leg A: flatness disabled | Leg B: flatness enabled (0.9) |
|--------|---------------------------|--------------------------------|
| Final hulls | 585 | 585 |
| Total vertices | 19,416 | 19,416 |
| Total triangles | 36,492 | 36,492 |
| Runtime | 1,051.8s (~17.5 min) | 1,042.4s (~17.4 min) |
| Validation | CLEAN | CLEAN |

Environment: Intel64 32 logical cores, Windows 11 (10.0.26200), Python 3.12.11, chitin 0.1.2, CoACD 1.0.11, concavity 0.05.

The 27-min figure in the Garden table above was measured on different hardware (M1 Pro, 16 cores) and is not comparable to these wall-clock numbers; only hull counts compare across machines, and even those aren't the same run (773,074 splats went to 1,725 hulls there, 585 here — different chitin/CoACD versions and pipeline state).

On this asset, at this chitin/CoACD version, the two legs produced identical hull counts and statistically indistinguishable runtimes. Stronger than the count match: `leg-a-disabled/colliders.phys` and `leg-b-enabled/colliders.phys` are byte-identical, both SHA-256 `fcd94fca4e417e8ae8d387b7f7ef1aac7bfc2d5640463d42b9cf842604e3a973` — proof, not inference, that no cell took the flat branch. The flatness detector did not fire (no octree cell in this scene crossed the 0.9 PCA-eigenvalue-ratio threshold), so it neither reduced hull count nor changed build time here. This does not reproduce the 1,725 -> 579 / 27 min -> 9 min figures previously stated in the top-level README; those numbers have no recorded run and should be treated as unverified until someone reruns the exact same chitin/CoACD version that produced them.

### Flatness follow-up: why it never fires, and what does

Environment: Intel64 32 logical cores, Windows 11 (10.0.26200), Python 3.12.11, chitin 0.1.2, CoACD 1.0.11, asset `mipnerf360_garden_crop_table.ply` (773,074 points), `--concavity 0.05`.

#### 1. Why the flatness detector never fires

Per-octree-cell planarity was measured directly (18 cells). The metric is the largest eigenvalue over the trace of the area-weighted second-moment matrix of unit face normals, the same quantity `is_flat_mesh` thresholds. It is bounded to [1/3, 1]: 1/3 is isotropic normals, 1 is a perfect plane.

With no flags: min 0.386, median 0.500, max 0.654. The default threshold is 0.9, so nothing fires, and no threshold works — 0.7 catches nothing, 0.6 catches 2 of 18, 0.5 catches 9 of 18 but 0.5 is barely above isotropic.

#### 2. A 2x2 ablation over `--thin-shell` and `--proximity-filter`

Planarity is the whole-cell metric above; triangles is the summed reconstructed triangle count across cells.

| arm | planarity median | planarity max | total triangles |
|---|---|---|---|
| neither | 0.500 | 0.654 | 239,883 |
| `--thin-shell` only | 0.490 | 0.637 | 489,304 |
| `--proximity-filter 5.0` only | 0.638 | 0.860 | 117,681 |
| both | 0.617 | 0.816 | 267,914 |

Proximity filtering alone is best on both axes. `--thin-shell` is net negative on this asset: planarity slightly worse than not using it, and triangle count doubles, because it extrudes a shell (`src/chitin/stages/filter.py:104-109`).

#### 3. End-to-end hull counts

| run | hulls | wall clock |
|---|---|---|
| baseline, no flags (before the change) | 585 | 17m32s |
| `--proximity-filter 5.0` explicit | 410 | 8m30s |
| no flags, after proximity became the point-cloud default | 409 | 9m34s |

410 and 409 come from the same effective configuration, run twice, producing different hull counts: the splat path is not bit-reproducible, and the cause is unexplained.

#### 4. Why fixing the flatness detector is low value here

A run with the real detector active (proximity filtering on) recorded hulls produced per octree cell. Hull count correlates with triangle count at +0.39 and with planarity essentially not at all:

- pearson(hull_count, whole-cell planarity) = -0.039
- pearson(hull_count, planar area fraction within 15 degrees) = -0.047
- pearson(hull_count, planar patch thickness) = +0.027

The most planar cell (0.860) produces 13 hulls. The most expensive cell produces 52 and is among the least planar (0.526).

The per-cell hull counts sum to 308, while the finished artifact has 410, because seam repair, dedup and culling run afterwards on the assembled hull set (`src/chitin/stages/splat.py:387-415`). The per-cell figures and the correlations above are sound, but the per-cell totals are not the final total and percentages of the final total should not be derived from them.

The earlier Garden run recorded above in this file describes one cell producing 285 hulls, a regime where flatness substitution would have mattered a great deal. The current octree partition does not produce such a cell, so this conclusion is tied to the current partitioning, on one asset.

### PLY mesh-path fix (Stanford Bunny)

PLY files with a face element were routed through the point-cloud/Poisson path: `adapters.ply.load_ply` never populated `AdapterResult.faces`, and `analyze._analyze_ply` hardcoded `face_count=None`, so `core.extract` (which branches on `result.faces is not None`) sent every `.ply` down the same path as an unstructured point cloud. Only `.obj/.stl/.off/.glb/.gltf` reached the mesh path, though `docs/usage.md` lists `.ply` alongside the mesh formats. The reader now parses the face element (ascii, binary little- and big-endian), fan-triangulates n-gons, and accepts both `vertex_indices` and `vertex_index`; PLY meshes take the mesh path.

Measured on `examples/utility-proof/assets/stanford-bunny/bunny.ply` (Stanford Bunny, 35,947 vertices, 69,451 faces declared):

| | before (point-cloud path) | after (mesh path) |
|---|---|---|
| coverage `covered_fraction` | 0.9386 (2,206 points uncovered) | 1.0 (0 uncovered) |
| hulls | 105 | 20 |
| mesh vertices used | 2,581 (Poisson reconstruction) | 35,947 (the authored mesh) |
| wall clock | 126.1s | 41.0s |
| `build_plan.pipeline` | `['parse','normal_estimation','reconstruct','decompose','coverage']` | `['parse','decompose','coverage']` |

Environment: Intel64 32 logical cores, Windows 11 (10.0.26200), Python 3.12.11, chitin 0.1.2, CoACD 1.0.11, default config.

Routing verification, same session:

| Asset | Face element | `face_count` | `pipeline_path` | `surface_proximity_filter` (resolved) |
|---|---|---|---|---|
| `bunny.ply` | yes | 69451 | mesh | 0.0 |
| `mipnerf360_garden_crop_table.ply` | no (3DGS splat) | None | splat | 5.0 |

Gaussian-splat and plain point-cloud PLY input take the splat path unchanged. The Garden measurements recorded earlier in this file (Mip-NeRF 360 Garden, Flatness detection, Flatness follow-up) used `mipnerf360_garden_crop_table.ply`, which has no face element, so this fix does not affect them.

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
