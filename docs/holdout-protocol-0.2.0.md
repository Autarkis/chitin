# Holdout Protocol — Policy 0.2.0

Frozen acceptance criteria for the Policy 0.2.0 holdout evaluation.
Declared before corpus selection to prevent outcome-peeking.

**Policy**: `POLICY_0_2_0` (grid_bits=20, ambiguity_fallback=True)
**Evaluator**: `scripts/evaluate_holdout.py --policy 0.2.0`
**Output**: `docs/holdout-results-0.2.0.json`

## Acceptance criteria

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Classification agreement | ≥ 99% | Same bar as 0.1.0 |
| Oracle agreement | ≥ 99.99% | Same bar as 0.1.0 |
| Zero invalid geometry | 0 | Same bar as 0.1.0 |
| Topology on disagreements | 100% face-set | The criterion 0.1.0 failed (0/114) |
| Stratified topology sample | ≥ 99% face-set | Same bar as 0.1.0 |

## Protocol

1. **Corpus must be disjoint.** No overlap with CI tier, external tier (spent
   on 0.1.0 holdout), or regression tier (#118 calibration). Enforced by
   `KNOWN_CORPUS_DIGESTS` in the evaluator.
2. **Evaluate exactly once.** The result is immutable. No re-evaluation on the
   same corpus, no parameter tuning after seeing results.
3. **Versioned output.** Results written to `docs/holdout-results-0.2.0.json`
   (separate from 0.1.0's `docs/holdout-results.json`).
4. **Full policy record.** The output JSON records all policy fields including
   `version`, `ambiguity_fallback`, `grid_bits`, `grid_scale`.

## Corpus status

No corpus selected. Selection follows this protocol freeze.

## Reference

- Policy 0.1.0 holdout result: `docs/f32-holdout-results.md` (FAIL, 0/114 topology)
- Policy 0.2.0 calibration: `docs/f32-gate-results.md` (114/114 classification + face-set agree)
- Regression corpus: `tests/fixtures/regression/manifest.json`
- Corpus digests: `tests/fixtures/traces/CORPUS_MANIFEST.md`
