# f32 Gate Protocol

Acceptance protocol for the f32 predicate gate (#108/#101). Frozen before holdout evaluation.

## Frozen policy

| Parameter | Value |
|-----------|-------|
| Grid bits | 20 (DEFAULT_POLICY) |
| Sweep range | 20–23 (regression tests) |
| Holdout evaluation | DEFAULT_POLICY only |

## Regression floors (CI)

These assert "at least as good as last measured." Applied to CI-tier corpus.

| Metric | Floor | Measured |
|--------|-------|----------|
| Classification agreement (per-fixture) | 90% | 93.7–100% |
| Clip topology agreement | 85% | ~93.7% aggregate |
| Cap topology agreement | 85% | ~93.7% aggregate |
| Oracle agreement (aggregate) | 85% | 99.0%+ |

## Final gate criteria

Applied to holdout evaluation (external-tier corpus). All must pass.

1. **Zero invalid outputs.** No open meshes, misoriented normals, or degenerate
   faces in any admitted clip/cap result.

2. **Zero unexplained skips or crashes.** Every skip must have a classified
   reason (zero-length normal, empty mesh). No segfault, hang, or timeout.

3. **Full topology on disagreements.** Every classification disagreement gets
   full clip+cap topology replay. The disagreement itself is acceptable if
   topology is preserved; the replay proves it.

4. **Stratified topology sample of agreements.** A deterministic risk-weighted
   sample of agreeing clips (biased toward large meshes and near-plane vertices)
   gets full topology replay. Minimum 10% of clips or 500, whichever is larger.

5. **Oracle comparison.** f32 vs C++ oracle Side on every clip with oracle data.
   Aggregate agreement must exceed 99%.

6. **Failure classification.** Every clip outside admission gets an explicit
   failure class: degenerate input, numerical boundary, unsupported topology.
   No "other" or unclassified failures.

## Holdout fixtures

| Fixture | Clips | Risk profile |
|---------|-------|-------------|
| t_shape | 1,054 | Medium mesh, moderate clip count |
| curved_pipe_quarter | 3,162 | Curved geometry, many near-plane vertices |
| h_shape | 20,954 | Largest fixture, stress test |

## Procedure

1. Run holdout evaluation once with frozen policy (grid_bits=20).
2. Record raw results immutably (no re-run after seeing results).
3. Apply final gate criteria above.
4. Issue PASS or FAIL verdict.
5. If PASS: close #108 and #101 with sha link.
6. If FAIL: document failure class, file remediation issue, do not close.

## Deferred measurements

**Candidate ordering preservation** is not measured by this gate. The f32
predicate gate validates classification, topology, and oracle agreement —
whether the f32 predicates produce equivalent geometry. Candidate ordering
(whether the search explores decompositions in the same sequence) is owned
by #95 (cost graph and deterministic ordering). This gate's CONDITIONAL PASS
does not assert ordering equivalence.
