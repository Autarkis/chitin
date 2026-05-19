# Utility Proof

Empirical validation that Chitin produces useful physics artifacts from real-world 3D assets.

## Question

Can Chitin take messy real 3D input and produce physics artifacts that are useful enough to ship, inspect, and reuse?

## Test cases

| Case | Dataset | Tests |
|------|---------|-------|
| Scanned object | Google Scanned Objects, YCB mug | Static mesh → convex hulls. Visual fit, hull count, sim stability |
| Scene / scan | Tanks and Temples — Barn | Scene-scale point cloud → walkable collision. Poisson reconstruction + decomposition |
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

## What success looks like

- Build completes without errors
- `chitin validate` passes clean
- Hull count is reasonable for the asset complexity (not 1, not 10000)
- Build time is under 60s for typical objects, under 5min for large scenes
- .phys file size is small relative to the source asset

Failures are expected and useful. A bad result becomes the roadmap: better defaults, better diagnostics, better preflight, or a documented limitation.

## Datasets

Assets are NOT checked into git. `datasets.toml` is the manifest; `download.py` fetches them. Each entry records URL, license, citation, format, and redistribution status.

## Notes

Requires Python 3.12 with chitin installed (`pip install -e .` from repo root).
