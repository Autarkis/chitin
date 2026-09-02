# f32 Predicate Disproof Gate — Results

**Issues**: #101, #108 (both open)
**Date**: 2026-09-02
**Policy tested**: `DEFAULT_POLICY` (grid_bits=20, classification_ulp_margin=0, intersection_snap_bits=20)
**Oracle**: CoACD 1.0.14, instrumented DLL, 5 C++ trace hooks (plane, clip, cost, MCTS, component state)

## Verdict: INCONCLUSIVE

Not a pass, conditional or otherwise. Not a failure either. #93 and #94 are
**not** unblocked by this measurement — G2/GPU work should not treat this
gate as cleared.

## What was measured

- 6 trace fixtures captured from the traced CoACD DLL (box, l_shape, t_shape,
  thin_panel, staircase, icosphere).
- f32 vs f64 vertex classification, replayed on 7,334 clips.
- Corpus-wide classification agreement: **93.7%** (461 disagreements).
- Worst fixture: staircase at **91.5%** (442 of the 461 disagreements).
- box, thin_panel, icosphere are already convex and produce zero clips —
  they contribute no signal.

## Why this is inconclusive, not a pass

1. **Tests did not enforce a threshold.** `test_trace_replay.py` printed
   agreement percentages but asserted nothing — the suite passed even at
   0% agreement.

2. **The classification comparison is not against the original mesh.** It
   reconstructs an approximate input mesh by concatenating each clip's
   positive/negative outputs, which duplicates cut vertices and loses vertex
   identity. The comparison is against a reconstruction, not CoACD's real
   intermediate state.

3. **No comparison against the C++ oracle's actual decisions.** Both sides
   of the measured comparison are Python (f64 NumPy vs f32 NumPy); neither
   is the traced C++ `Side`/`CutSide` classification itself.

4. **"Classification agreement implies topology agreement" is unsupported.**
   Even with identical per-vertex signs, f32 intersections, snapping,
   deduplication, loop construction, and winding can still diverge. One
   matching data point (l_shape, 98.5% both) is not a proof this holds in
   general.

5. **staircase and t_shape only got classification replay**, not full
   clip+cap topology replay — the topology comparison is O(n²) in the Python
   harness and those two fixtures produce 10K–26K vertices per clip after
   CoACD's voxel preprocessing, making full replay infeasible in test time.

6. **Only `DEFAULT_POLICY` was tested.** No sweep across `grid_bits`,
   `classification_ulp_margin`, or `intersection_snap_bits` against the
   trace corpus.

7. **The corpus barely exercises decomposition.** Only 3 of 6 fixtures
   produce clips at all, and of the 7,334 clips measured, staircase alone
   accounts for 5,226 — the result is dominated by one fixture.

8. **The epsilon-tolerance checker missed bare numeric comparisons**, e.g.
   `norm < 1e-15` written as a literal rather than threaded through
   `QuantizationPolicy` — a gap in the tooling meant to catch absolute
   world-unit epsilons slipping into grid-relative code.

## What's been done to address this

- C++ instrumentation extended to record the original input mesh plus each
  triangle's actual `Side` decision at `Clip()` entry.
- `save_trace`/`load_saved_trace` extended to persist this oracle data.
- Tests now assert thresholds (classification ≥ 90%, clip/cap topology ≥ 85%)
  instead of printing and passing unconditionally.
- Full topology replay work started for staircase/t_shape (previously
  classification-only).
- A policy sweep test was added (grid_bits 12, 16, 20, 22).
- An oracle comparison test was added, skipping gracefully on v1 (pre-oracle)
  traces.
- The epsilon checker was expanded to catch bare numeric comparisons, not
  just named tolerance assignments.

## What remains before the gate can pass

- Regenerate the trace corpus with the v2 instrumented DLL (captures input
  mesh + oracle sides) — the saved corpus on disk is still v1.
- Run the oracle comparison and measure f32 vs the C++ oracle's actual
  decisions, not f64-NumPy-vs-f32-NumPy.
- Finish and run full clip+cap topology replay on staircase and t_shape, and
  report the measured rates (not just classification).
- Sweep policies across the full corpus and characterize the envelope.
- Define final pass thresholds from the measured data — the current 90%/85%
  values are regression floors pinned to today's numbers, not derived
  acceptance criteria.
- Add nonconvex thin and curved fixtures that force multi-hull decomposition;
  the current corpus is convex-dominated (half the fixtures produce zero
  clips) and staircase-dominated among the rest.
- If exact f32/f64 agreement turns out to be unreachable, document
  quantitatively what diverges and show end-to-end candidate-ordering and
  final-collider quality are preserved despite it.

## Reproduction

```bash
# Capture corpus (requires the traced DLL deployed)
python scripts/capture_trace_corpus.py

# Run gate tests (uses saved corpus, no DLL needed)
python -m pytest tests/test_trace_replay.py -v -s
```
