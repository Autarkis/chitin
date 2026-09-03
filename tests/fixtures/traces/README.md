# CoACD trace corpus

Traces of CoACD's internal geometric predicate decisions, captured with a
v2-instrumented build of CoACD v1.0.14 in stream v3 format. Used by the f32
predicate gate replay tests.

## Two tiers

**CI tier** (tracked in git, ~45 MB total): box, icosphere, thin_panel,
l_shape, thin_u_channel, cross_bracket, staircase. Replayed on every CI run.

**External tier** (gitignored, ~17.4 GB total): t_shape, curved_pipe_quarter,
h_shape. Too large for git. Used for holdout evaluation and offline replay.
See `CORPUS_MANIFEST.md` for digests.

## Regenerating

Requires the traced DLL (see `tools/BUILD_CONTRACT.md`):

```
python scripts/capture_trace_corpus.py
```

## Integrity

CI verifies corpus digests against `CORPUS_MANIFEST.md`. Missing or corrupted
CI-tier fixtures fail the gate when `CHITIN_GATE_FINAL=1`.
