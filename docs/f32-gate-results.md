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
robust path would execute extraordinarily rarely.

### Path to Policy 0.2.0

1. Record 114 known failures as regression cases.
2. Design a filtered predicate: ordinary f32 for almost every vertex, explicit
   floating-point error bound, deterministic fixed-point or compensated evaluation
   only inside the ambiguous band.
3. Use only the calibration corpus for development (holdout is spent).
4. Evaluate Policy 0.2.0 against a new unseen holdout.

Intersection-coordinate drift (the coordinate-only failures) does not block the
architecture; 114 connectivity changes do.

### What passed

1. **Oracle agreement**: 100.00% (166,659,465/166,659,512 vertices).
2. **Zero invalid outputs**: no open/misoriented/degenerate geometry.
3. **Zero unexplained skips**: 108 across 25,170 clips, all classified.
4. **Classification agreement**: 99.5%+ across all fixtures.
5. **Topology on agreeing clips**: 100% face-set preservation (3,076/3,076 sampled).
6. **Full topology on all 114 disagreement clips**: replayed and classified.

## Corpus

**CI tier** (7 fixtures, ~45 MB, tracked in git): box, icosphere, thin_panel,
l_shape, thin_u_channel, cross_bracket, staircase.

**External tier** (3 fixtures, ~17.4 GB, holdout): t_shape, curved_pipe_quarter, h_shape.

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
