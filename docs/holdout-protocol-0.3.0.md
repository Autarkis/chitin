# Holdout Protocol — Policy 0.3.0

Frozen acceptance criteria for the Policy 0.3.0 holdout evaluation.
Declared before corpus capture to prevent outcome-peeking.

**Policy**: `POLICY_0_3_0` (canonical_f32_inputs=True, ambiguity_fallback=True)
**Evaluator**: `scripts/evaluate_holdout.py --policy 0.3.0`
**Output**: `docs/holdout-results-0.3.0.json`

## What's new in 0.3

- **Canonical f32 input contract** (`docs/policy-0.3-input-contract.md`, #122): vertices,
  plane point, and plane normal are cast to f32 then back to f64 before evaluation.
  Sign information that exists only in higher-precision source inputs is outside the
  contract — evaluation operates on the canonical values, not the raw source values.
- **Precision-loss categorization.** The evaluator's `categorize_disagreements()` helper
  splits every classifier disagreement into two categories (not necessarily disjoint — a
  clip can exhibit both source precision loss and genuine arithmetic disagreement):
  - **Source precision loss** — `raw_ref != canon_ref`. The exact sign of the raw source
    input differs from the exact sign of the canonicalized f32 input. Reported, not a
    classifier failure. A clip with precision loss may also have a genuine arithmetic
    disagreement.
  - **Genuine f32 arithmetic disagreement** — `canon_ref != canon_cand`. The policy's
    classification disagrees with the f64 reference classification over the *canonical*
    f32 inputs. This is the 0.3 contract's true failure category. Note: the production
    evaluator computes the reference classification using f64 arithmetic, not exact
    rational arithmetic. Exact-rational behavior is independently verified by the
    adversarial predicate harness (`tests/test_adversarial_predicates.py`).
- **Stratum-aware evaluation.** The corpus has two independent strata (below). Acceptance
  criteria are computed and judged separately per stratum; there is no pooled criterion.

## Acceptance criteria (per stratum)

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Classification agreement | ≥ 99% | Same bar as 0.1/0.2 |
| Oracle agreement | ≥ 99.99% | Same bar as 0.1/0.2 |
| Zero invalid geometry (NaN/Inf) | 0 | Hard floor |
| Topology on disagreements | 100% face-set | Same bar as 0.2 |
| Stratified topology sample | ≥ 99% face-set | Same bar as 0.2 |
| Precision-loss clips | Reported only | Informational; a clip may have both precision loss and genuine disagreement |
| Genuine arithmetic disagreement | Counted as failures | Counted against classification agreement; the 0.3 contract's true failure category |

Every criterion above is evaluated independently within each stratum. Overall verdict:
**PASS iff every criterion PASSes in both strata; any per-stratum FAIL is an overall FAIL.**

## Strata

- **Ordinary.** Unit-scale geometry (same regime as the 0.1.0/0.2.0 corpora). Exercises
  the standard f32 precision regime — f32 ULP well below the ambiguity band at these
  coordinate magnitudes.
- **Large-offset.** The same fixture geometry as the ordinary stratum, translated to
  coordinate offsets `[1e7, 5e6, 3e6]`. At this magnitude f32 spacing (ULP) is
  approximately 1, 0.5, and 0.25 per component — comparable to sub-unit geometric
  features, so precision loss is structural rather than incidental. Fixtures are matched pairs with the ordinary
  stratum: identical source topology, identical relative geometry, only the coordinate
  origin differs. Because CoACD captures each fixture independently, decomposition
  behavior (clip planes, execution path, part count) may diverge under translation even
  when source topology is identical. A matched pair therefore controls source topology
  and relative geometry while intentionally allowing decomposition behavior to diverge.
  Verifying strict plane-by-plane invariance under translation would require a separate
  replay experiment using shared translated planes.
- Each stratum is evaluated independently against the full criteria table. There is no
  combined/pooled metric — a stratum cannot compensate for the other's shortfall.

## Corpus requirements

- ≥ 30,000 clips per stratum (60,000 clips total floor).
- ≥ 3 distinct topology families per stratum.
- Large-offset fixtures must be geometry-matched to their ordinary-stratum counterpart
  (same source mesh, translated coordinates only; clip planes are independently captured
  and may differ).
- Holdout fixtures must not overlap the CI tier, the external tier, or fixtures already
  spent on the 0.1.0 or 0.2.0 holdouts. Enforced by `KNOWN_CORPUS_DIGESTS` in the evaluator.

## Protocol

1. **Corpus must be disjoint.** No overlap with CI tier, external tier, regression tier,
   or prior (0.1.0/0.2.0) holdout fixtures.
2. **Chain of custody.** Evaluation requires a `capture-record.json` produced by the safe
   capture CLI; the evaluator verifies manifest and per-fixture trace digests.
3. **Evaluate exactly once, per stratum.** Results are immutable. The canonical output
   path (`docs/holdout-results-0.3.0.json`) cannot be overwritten, even with `--force`.
   No re-evaluation on the same corpus, no parameter tuning after seeing results.
4. **Versioned output.** Results written to `docs/holdout-results-0.3.0.json`, separate
   from 0.1.0's and 0.2.0's result files. Output includes both strata's metrics and the
   precision-loss / genuine-arithmetic-disagreement breakdown.
5. **Full policy record.** Output JSON records all policy fields including `version`,
   `canonical_f32_inputs`, `ambiguity_fallback`, `grid_bits`, `grid_scale`.
6. **Traced-DLL identity.** The capture CLI verifies the deployed CoACD shared library's
   hash against the manifest's `traced_coacd.dll_digest`.

## Tooling

- **Selection manifest**: `docs/holdout-corpus-0.3.0.json` — committed before capture,
  declares fixture names, stratum assignment, source-geometry digests, capture config,
  and traced-CoACD identity.
- **Capture**: `scripts/capture_holdout_corpus.py --manifest <path> --output-dir <dir>` —
  verifies source-geometry digests, refuses to overwrite, records trace digests in
  `capture-record.json`.
- **Evaluation**: `scripts/evaluate_holdout.py --policy 0.3.0 --corpus-manifest <path>` —
  manifest-driven fixture loading, stratum-aware metrics via `categorize_disagreements()`,
  complete digest rejection, overwrite protection, machine-computed per-stratum and
  overall PASS/FAIL verdict.
- Output is immutable once written, per the protocol above.

## Reference

- Policy 0.3.0 input contract: `docs/policy-0.3-input-contract.md` (#122)
- Policy 0.2.0 holdout result: `docs/holdout-protocol-0.2.0.md` (FAIL, `0d56ee4`)
- Policy 0.2.0 failure diagnosis: `docs/holdout-0.2.0-failure-analysis.md`
- On-plane oracle contract (#123): tie-breaking convention, not a precision defect
- Corpus digests: `tests/fixtures/traces/CORPUS_MANIFEST.md`
- Safe capture CLI: `scripts/capture_holdout_corpus.py`

## Outcome (post-evaluation record)

**Verdict: PASS** — evaluated once at `57fe416`, result immutable at
`docs/holdout-results-0.3.0.json`.

| Metric | Ordinary | Large-offset |
|--------|----------|--------------|
| Clips captured | 309,779 | 275,282 |
| Clips replayed | 308,708 | 274,480 |
| Clips skipped | 1,071 | 802 |
| Classification agreement | 100.0% | 100.0% |
| Oracle agreement | 99.99999% | 99.99998% |
| Invalid geometry | 0 | 0 |
| Precision-loss clips | 752 | 201 |
| Genuine arithmetic disagreements | 0 | 0 |

Zero genuine arithmetic disagreements across 583,188 replayed clips. The f32
canonical contract holds: after canonicalization, f32 classification matches
f64 reference on every clip in both strata.

This PASS supports Policy 0.3.0 as Chitin's current `DEFAULT_POLICY`. It closes
the plane-classification portability gate; clip/cap implementation and
end-to-end WebGPU admission remain separate gates.

Precision-loss composition: ordinary-stratum matched fixtures contribute 218
precision-loss clips; `interlocked_frame` (ordinary-only) contributes 534.
Large-offset matched fixtures contribute 201. Precision loss is reported, not
counted as failure per the 0.3 contract.

Result digest: `7663195704846a9533da776d4275dc39fe5bea74316f9397cf3d616597370b7b`.
