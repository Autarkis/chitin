# CoACD trace corpus

Traces of CoACD's internal geometric predicate decisions, captured with a
v2-instrumented build of CoACD v1.0.14 in stream v3 format. Used by the f32
predicate gate replay tests.

## Corpus layout

**CI tier** (tracked in git, ~45 MB total): box, icosphere, thin_panel,
l_shape, thin_u_channel, cross_bracket, staircase. Replayed on every CI run.

**External tier** (gitignored, ~17.4 GB total): t_shape, curved_pipe_quarter,
h_shape. Too large for git. The original Policy 0.1.0 holdout is spent; these
traces remain available for offline replay.

**Regression tier** (tracked in git): the 114 Policy 0.1.0 failures and 14
Policy 0.2.0 diagnostic clips promoted from spent holdouts. These are reusable
test inputs, not unseen evaluation data.

Policy 0.2.0 and Policy 0.3.0 holdout corpora are gitignored and spent. Their
selection manifests, digests, and immutable outcomes remain tracked. See
`CORPUS_MANIFEST.md` for the complete inventory and chain of custody.

## Regenerating

Requires the traced DLL (see `tools/BUILD_CONTRACT.md`):

```
python scripts/capture_trace_corpus.py
```

## Integrity

CI verifies corpus digests against `CORPUS_MANIFEST.md`. Missing or corrupted
CI-tier fixtures fail the gate when `CHITIN_GATE_FINAL=1`.
