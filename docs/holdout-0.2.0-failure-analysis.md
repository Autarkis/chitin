# Policy 0.2.0 Holdout Failure Analysis

Evaluator commit: `c99c4a1`. Result commit: `0d56ee4` (2026-09-04).
Results digest: `39f87937d290c0b4cff1c676e8feb3efe8b130e3778e37967cd4ff58e849a7f7`.
Verdict: **FAIL** — three of five acceptance criteria missed.

## Two independent failure clusters

### Cluster 1: f32 grid-boundary classification failures (14 clips)

| Fixture | Failing clips | Component |
|---------|---------------|-----------|
| twisted_notched_column | 124, 125, 151, 152, 187, 188, 277, 278, 511, 512, 673, 674 | 0 |
| multiscale_shard_cluster | 825, 3848 | 0, 3 |

**Pattern:** all failures are consecutive pairs on centroid-axis planes. The twisted_notched_column
failures are 6 pairs at planes z=0.5, z=0.33, z=-0.33 (grid-aligned values). The shard cluster
failures are on x and z axes at non-round but grid-boundary-adjacent values.

**Mechanism:** f32 quantization (`grid_bits=20`) maps the vertex coordinate to a different grid
cell than f64, so f32 and f64 classify the vertex to opposite sides of the splitting plane. The
clip result (which side each vertex belongs to) therefore disagrees, producing:
- Classification disagreement (fails the classification check)
- Infinite residual (no valid intersection to measure — fails the zero-invalid-geometry check; 14 Inf, 0 NaN)
- Face-set topology disagreement (different faces on each side — fails the disagree-topology check)

**Impact:** accounts for all 14 infinite residuals (0 NaN) and the 0% disagree-topology rate.

**Diagnostic set:** 14 `.npz` files in `tests/fixtures/traces/holdout_failures_0_2_0/` with
vertices, faces, oracle sides, plane normal, and plane offset. Manifest at
`tests/fixtures/traces/holdout_failures_0_2_0/manifest.json`.

### Cluster 2: oracle on-plane tie-breaking disagreement (1,646 vertices)

| Fixture | Oracle disagrees | Dominant plane |
|---------|-----------------|----------------|
| skewed_rectangular_torus | 1,414 | y=0 (1,274), y=±0 (112), z=±0 (16), others (12) |
| twisted_notched_column | 230 | z=0 (122), z=±0.33 (107), y=0.016 (1) |
| oblique_gear_prism | 1 | — |
| multiscale_shard_cluster | 1 | — |

**Pattern:** every disagreeing vertex sits exactly on the clipping plane (dot product 10^-17 to
10^-7). The torus generator places ring-0 vertices at y=0; the column generator places station
vertices at even z-offsets including z=0 and z=±1/3.

**Mechanism:** the C++ CoACD oracle returns `side=0` for on-plane vertices. The Python f32
classifier computes a nonzero (but machine-epsilon) dot product and returns `side=+1`. This is a
**tie-breaking contract mismatch**, not a precision defect — the two implementations use different
conventions for the degenerate case.

**Impact:** accounts for the oracle rate dropping to 99.964% (below the 99.99% threshold). Does
not cause classification failures — f32 and f64 agree with each other on every torus clip, just
not with the C++ oracle.

## Separation

The two clusters are independent:
- Cluster 1 fails classification (f32 ≠ f64). Cluster 2 passes classification (f32 = f64).
- Cluster 1 is a quantization-boundary issue. Cluster 2 is a tie-breaking convention issue.
- Fixing either does not automatically fix the other.

Neither cluster is explained by vertex welding (#120).

## Forward

- Policy 0.2.0 is spent. No re-evaluation against this holdout.
- Fixes become Policy 0.3.0 with newly frozen criteria and a fresh holdout corpus.
- Cluster 1 fix: grid-boundary handling in the f32 classifier (ULP margin, or
  consistent rounding at grid boundaries).
- Cluster 2 fix: align on-plane convention — either the Python classifier adopts the
  C++ side=0 convention, or the oracle comparison treats side=0 as matching side=+1
  (since CoACD's convention is that on-plane vertices go to the positive halfspace).
