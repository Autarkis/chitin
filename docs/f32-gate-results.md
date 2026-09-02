# f32 Predicate Disproof Gate — Results

**Issues**: #101, #108
**Date**: 2026-09-02
**Policy tested**: `DEFAULT_POLICY` (grid_bits=20, classification_ulp_margin=0, intersection_snap_bits=20)
**Oracle**: CoACD 1.0.14, instrumented DLL (v2), 5 C++ trace hooks
**Build contract**: `tools/BUILD_CONTRACT.md`

## Verdict: CONDITIONAL PASS

See `docs/f32-holdout-results.md` for the immutable holdout evaluation record.

### Summary

f32 vertex classification is validated: 99.5%+ agreement with both f64 reference and
C++ oracle across 25,062 clips and 166M vertices in a genuinely unseen holdout. All
disagreements are near-plane vertices (|dot| < 1e-6). Oracle agreement is effectively
100%.

Clip-mesh intersection coordinates carry sub-mm positional drift under f32 arithmetic
(87–98% topology agreement depending on mesh complexity). This is a known, classified
limitation — not a classification error and not a collider-fitness defect.

### What was resolved since the inconclusive verdict

Every item from the original "What remains" list:

1. **v2 trace corpus regenerated.** All 10 fixtures captured with the v2 instrumented
   DLL (input mesh + oracle sides). Stream v3 format (concatenated arrays, ~20 npz
   entries per fixture instead of 167k files).

2. **Oracle comparison measured.** f32 vs C++ `Side()` decisions: 100.00% agreement
   across 166,659,512 vertices, 47 near-plane disagreements (max |dot| 7.2e-7).

3. **Full topology replay completed** on holdout fixtures including t_shape and
   h_shape (20,954 clips). Stratified risk-weighted samples of 10%+ per fixture.

4. **Policy sweep across corpus.** grid_bits 20–23 swept on all CI-tier fixtures
   with regression floors.

5. **Nonconvex fixtures added.** thin_u_channel, cross_bracket, curved_pipe_quarter
   — all force multi-hull decomposition with curved/thin geometry.

6. **Failure classification complete.** All topology disagreements classified as
   intersection-point positional drift, not classification or topological errors.

### Conditions

1. Clip topology is a noted limitation (intersection-point f32 drift), not a defect.
2. h_shape (87% clip topology) is the extreme stress test floor; other fixtures 90–98%.
3. #93 (GPU geometry core) may proceed with the understanding that vertex classification
   is safe and intersection coordinates carry sub-mm drift at grid_bits=20.

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

# Run with integrity check
CHITIN_GATE_FINAL=1 python -m pytest tests/test_trace_replay.py -v -s
```
