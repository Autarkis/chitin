# f32 Predicate Disproof Gate — Results

**Issues**: #101, #108
**Date**: 2026-09-02 (evaluated), 2026-09-02 (verdict updated)
**Policy tested**: `DEFAULT_POLICY` (grid_bits=20, classification_ulp_margin=0, intersection_snap_bits=20)
**Oracle**: CoACD 1.0.14, instrumented DLL (v2), 5 C++ trace hooks
**Build contract**: `tools/BUILD_CONTRACT.md`
**Evaluator**: `scripts/evaluate_holdout.py`

## Verdict: FAIL — Policy 0.1.0

See `docs/f32-holdout-results.md` for the immutable holdout evaluation record,
and `docs/holdout-results.json` for the machine-readable evidence.

### Ruling

Every classification disagreement changed clip connectivity:

| Fixture | Classification disagreements | Face-set failures among them |
|---|---:|---:|
| t_shape | 2 | 2 |
| curved_pipe_quarter | 0 | 0 |
| h_shape | 112 | 112 |

The frozen protocol requires topology preservation when classification disagrees.
That condition failed 114/114 times. The gate does not pass.

### Diagnosis

The clip implementation is structurally sound whenever classification agrees:

- t_shape: 500/500 agreeing samples preserve face sets.
- curved_pipe: 500/500.
- h_shape: 2,076/2,076.

The remaining problem is the rare near-plane predicate decision. Only 47 vertex
decisions out of 166M differed from the C++ oracle, all at |dot| < 7e-7. A slow
robust path would rarely execute.

### Refined diagnosis (#119)

Per-clip first-divergence classification (`docs/divergence-report.json`) isolates
the cause to **grid quantization**, not raw f32 rounding:

| Variant | Classification disagree | Face-set disagree |
|---------|------------------------:|------------------:|
| raw f32 (no grid) | 0/114 | 0/114 |
| grid, no snap | 114/114 | 114/114 |
| Policy 0.1.0 (grid + snap) | 114/114 | 114/114 |

`normalize_to_grid` maps vertices onto a 2^20 integer grid. For these 114 clips,
1-2 vertices per clip (108 clips: 1 vertex; 6 clips: 2 vertices) land on the
wrong side of the plane after quantization — all at |dot| < 7e-7 in the original
frame, collapsing to exactly zero in the grid frame.

Raw f32 arithmetic without the grid has zero divergence from f64 on all 114 clips.
Intersection snapping contributes no additional classification error beyond what
the grid already introduces.

### Path to Policy 0.2.0

1. Record 114 known failures as regression cases (#118) — **done**.
2. Classify first divergence per clip (#119) — **done**: all 114 diverge at
   classification due to grid quantization.
3. Design and implement a filtered predicate (#115) — **done**: unquantized-f32
   fallback for vertices within the grid quantization error bound.
4. Use only the calibration corpus for development (holdout is spent).
5. Evaluate Policy 0.2.0 against a new unseen holdout.

Intersection-coordinate drift (the coordinate-only failures) does not block the
architecture; 114 connectivity changes do.

## Policy 0.2.0 — Filtered Grid-Boundary Classifier

**Issue**: #115
**Date**: 2026-09-03
**Policy**: `POLICY_0_2_0` (grid_bits=20, classification_ulp_margin=0, intersection_snap_bits=20, ambiguity_fallback=True)

### Mechanism

Grid quantization (`normalize_to_grid`) maps vertices to a 2^20 integer grid,
introducing ±0.5 per-coordinate rounding error. The worst-case dot-product error
for a vertex is the L1 norm of the grid-frame normal:

    bound = Σ|grid_normal_i|

where `grid_normal = plane_normal × scale_factor`. This bound is derived from
first principles: both vertex and plane point are quantized (±0.5 each → ±1.0
worst-case difference per component), and the error propagates through the dot
product proportional to each normal component.

For vertices where `|grid_dot| > bound`, the grid classification is exact — no
quantization error can flip the sign. These go through the fast path (grid-only,
GPU-portable).

For vertices where `|grid_dot| ≤ bound`, the vertex is within the quantization
ambiguity band. These are reclassified using unquantized f32 arithmetic on the
original world-space coordinates, which #119 proved has zero divergence from f64
on all 114 regression clips. This fallback is deterministic and GPU-portable
(WGSL `f32` arithmetic, no `f64` dependency).

### Calibration result

| Corpus | Classification agrees | Face-set agrees |
|--------|----------------------:|----------------:|
| CI tier (7 fixtures, 11,550 clips) | 100% | 100% |
| Regression (114 clips, #118) | 114/114 | 114/114 |

Policy 0.1.0 scores on the same regression corpus: 0/114 classification, 0/114 face-set.

### WGSL portability

The filtered predicate uses WGSL-portable semantics:
1. Compute grid-frame dot product (integer grid → f32 arithmetic).
2. Compute bound as `abs(grid_n.x) + abs(grid_n.y) + abs(grid_n.z)`.
3. If `abs(dot) > bound`: classify from grid dot (fast path).
4. If `abs(dot) <= bound`: classify from unquantized f32 dot on original coordinates.

No f64, no host readback, no branching on external state. The geometry stays
GPU-resident.

### Status

Calibration passed. Holdout evaluation pending — requires a genuinely new unseen
corpus (the #101 holdout is spent, and the #118 regression clips are now calibration).

### What passed (Policy 0.1.0)

1. **Oracle agreement**: 100.00% (166,659,465/166,659,512 vertices).
2. **Zero invalid outputs**: no open/misoriented/degenerate geometry.
3. **Zero unexplained skips**: 108 across 25,170 clips, all classified.
4. **Classification agreement**: 99.5%+ across all fixtures.
5. **Topology on agreeing clips**: 100% face-set preservation (3,076/3,076 sampled).
6. **Full topology on all 114 disagreement clips**: replayed and classified.

## Corpus

**CI tier** (7 fixtures, ~45 MB, tracked in git): box, icosphere, thin_panel,
l_shape, thin_u_channel, cross_bracket, staircase.

**External tier** (3 fixtures, ~17.4 GB, holdout — spent): t_shape,
curved_pipe_quarter, h_shape.

**Regression tier** (#118, ~12 MB, tracked in git): 114 clips extracted from
the spent holdout where classification disagreement changed connectivity.
Per-clip `.npz` at `tests/fixtures/regression/`. Regression test:
`tests/test_regression_corpus.py`.

Digests: `tests/fixtures/traces/CORPUS_MANIFEST.md`.

## Reproduction

```bash
# Build traced DLL (requires MSVC + CMake)
tools/build-traced-coacd.sh

# Capture corpus (requires traced DLL deployed)
python scripts/capture_trace_corpus.py

# Run gate tests (uses saved corpus, no DLL needed)
python -m pytest tests/test_trace_replay.py -v -s

# Run holdout evaluation
python scripts/evaluate_holdout.py

# Run with integrity check
CHITIN_GATE_FINAL=1 python -m pytest tests/test_trace_replay.py -v -s
```
