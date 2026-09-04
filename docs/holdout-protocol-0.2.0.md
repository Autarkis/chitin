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
| Zero invalid geometry | 0 | NaN/Inf clip residuals; structural cap-loop welding (#120) is out of scope |
| Topology on disagreements | 100% face-set | The criterion 0.1.0 failed (0/114) |
| Stratified topology sample | ≥ 99% face-set | Same bar as 0.1.0 |

## Protocol

1. **Corpus must be disjoint.** No overlap with CI tier, external tier (spent
   on 0.1.0 holdout), or regression tier (#118 calibration). Enforced by
   `KNOWN_CORPUS_DIGESTS` in the evaluator (11 known digests).
2. **Chain of custody.** Evaluation requires a `capture-record.json` produced by
   the safe capture CLI. The evaluator verifies the record's manifest digest
   matches the declared manifest, and each fixture's `arrays.npz` digest matches
   its recorded trace digest. Trace directories replaced after capture are rejected.
3. **Evaluate exactly once.** The result is immutable. The canonical output path
   (`docs/holdout-results-0.2.0.json`) cannot be overwritten, even with `--force`.
   No re-evaluation on the same corpus, no parameter tuning after seeing results.
4. **Versioned output.** Results written to `docs/holdout-results-0.2.0.json`
   (separate from 0.1.0's `docs/holdout-results.json`).
5. **Full policy record.** The output JSON records all policy fields including
   `version`, `ambiguity_fallback`, `grid_bits`, `grid_scale`.
6. **Traced-DLL identity.** The capture CLI verifies the deployed CoACD shared
   library's hash against the manifest's `traced_coacd.dll_digest`.

## Tooling

- **Selection manifest**: `docs/holdout-corpus-0.2.0.json` — committed before capture,
  declares fixture names, source-geometry digests, capture config, and traced-CoACD identity.
  Declares four holdout fixtures with strata, rationale, parameters, and source digests.
- **Safe capture**: `scripts/capture_holdout_corpus.py --manifest <path> --output-dir <dir>` —
  verifies source-geometry digests, refuses to overwrite, records trace digests in
  `capture-record.json`.
- **Evaluation**: `scripts/evaluate_holdout.py --policy 0.2.0 --corpus-manifest <path>` —
  manifest-driven fixture loading, complete digest rejection (11 known corpora),
  overwrite protection, machine-computed PASS/FAIL verdict.

## Corpus status

Four fixtures selected: `oblique_gear_prism` (concave-oblique), `twisted_notched_column`
(swept-nonparallel), `skewed_rectangular_torus` (genus-one), `multiscale_shard_cluster`
(multi-component-scale). 60,810 clips captured and evaluated.

## Result — FAIL

Evaluated 2026-09-04 (`0d56ee4`). Immutable evidence: `docs/holdout-results-0.2.0.json`.

| Criterion | Threshold | Measured | Result |
|-----------|-----------|----------|--------|
| Classification agreement | ≥ 99% | 99.977% | PASS |
| Oracle agreement | ≥ 99.99% | 99.964% | **FAIL** |
| Zero invalid geometry (NaN/Inf) | 0 | 14 Inf | **FAIL** |
| Topology on disagreements | 100% face-set | 0% (14 clips) | **FAIL** |
| Stratified topology sample | ≥ 99% | 99.77% | PASS |

### Diagnosis

Two independent failure clusters:

**Cluster 1 — 14 classification failures** (twisted_notched_column: 12, multiscale_shard_cluster: 2).
All on grid-boundary planes where f32 quantization moves vertices to a different side than f64.
Failures come in consecutive pairs on centroid-axis planes (z=0.5, z=1/3). Produces the 14
Inf residuals (14 Inf, 0 NaN) and the 0% disagree-topology. One causal cluster, one fix needed.
Diagnostic clips extracted to `tests/fixtures/traces/holdout_failures_0_2_0/`.

**Cluster 2 — 1,646 oracle vertex disagreements** (skewed_rectangular_torus: 1,414,
twisted_notched_column: 230, oblique_gear_prism: 1, multiscale_shard_cluster: 1).
All near-plane (max dot product < 10^-6). Vertices sit exactly on the clipping plane
(dot ≈ machine epsilon). C++ CoACD returns side=0 (on-plane), Python f32 classifier
returns side=+1 (positive). On-plane tie-breaking contract mismatch, not a classifier
defect. Does not cause classification failures — only oracle-rate failure.

Neither cluster is vertex welding (#120).

## Reference

- Policy 0.1.0 holdout result: `docs/f32-holdout-results.md` (FAIL, 0/114 topology)
- Policy 0.2.0 calibration: `docs/f32-gate-results.md` (114/114 classification + face-set agree)
- Regression corpus: `tests/fixtures/regression/manifest.json`
- Corpus digests: `tests/fixtures/traces/CORPUS_MANIFEST.md`
- Selection manifest schema: `docs/holdout-corpus-0.2.0.json`
- Safe capture CLI: `scripts/capture_holdout_corpus.py`
