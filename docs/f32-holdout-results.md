# f32 Holdout Evaluation Results

Immutable evidence record. Generated from `docs/holdout-results.json`.

**Verdict: FAIL — Policy 0.1.0** (see `docs/f32-gate-results.md`)

**Policy:** Policy 0.1.0 (the default when evaluated; grid_bits=20)
**Date:** 2026-09-02
**DLL:** traced CoACD v1.0.14 (`dd295d37...51853d3`)
**Evaluator:** `scripts/evaluate_holdout.py`

## Decisive finding

Every classification disagreement changed clip connectivity (114/114):

| Fixture | Classification disagreements | Face-set failures among them |
|---|---:|---:|
| t_shape | 2 | 2 |
| curved_pipe_quarter | 0 | 0 |
| h_shape | 112 | 112 |

Sampled classification-agreeing clips preserve face sets at 100% (3,076/3,076).
The predicate is structurally sound; the failure is the near-plane boundary decision.

## t_shape (1,054 clips, 2 parts)

| Metric | Value |
|---|---|
| Classification | 99.81% (1,051/1,053) |
| Classification disagreements | 2 |
| Skipped clips | 1 |
| Oracle agreement | 100.00% (16,607,677/16,607,677) |
| Topology sample | 502 clips (2 disagree + 500 sampled) |
| Face-set agreement | 99.60% (500/502) |
| Coordinate agreement | 92.03% (462/502) |
| Coord-only failures | 38 |
| Both failures | 2 |
| Cap topology | 99.60% (500/502) |

Intersection error (500 finite residuals, 2 null: 2 inf):
p50=3.79e-6, p95=1.01e-4, p99=6.43e-4, max=6.90e-4. Scale-relative: p50=0.043, p95=1.14, p99=7.26, max=7.79.

## curved_pipe_quarter (3,162 clips, 4 parts)

| Metric | Value |
|---|---|
| Classification | 100.00% (3,130/3,130) |
| Classification disagreements | 0 |
| Skipped clips | 32 |
| Oracle agreement | 100.00% (24,942,240/24,942,240) |
| Topology sample | 500 clips (0 disagree + 500 sampled) |
| Face-set agreement | 100.00% (500/500) |
| Coordinate agreement | 99.60% (498/500) |
| Coord-only failures | 2 |
| Both failures | 0 |
| Cap topology | 100.00% (500/500) |

Intersection error (absolute):
p50=3.38e-6, p95=1.07e-5, p99=2.90e-5, max=1.36e-4. Scale-relative: p50=0.048, p95=0.15, p99=0.41, max=1.92.

## h_shape (20,954 clips, 6 parts)

| Metric | Value |
|---|---|
| Classification | 99.46% (20,767/20,879) |
| Classification disagreements | 112 |
| Skipped clips | 75 |
| Oracle agreement | 99.9999% (125,109,548/125,109,595) |
| Topology sample | 2,188 clips (112 disagree + 2,076 sampled) |
| Face-set agreement | 94.88% (2,076/2,188) |
| Coordinate agreement | 87.52% (1,915/2,188) |
| Coord-only failures | 161 |
| Both failures | 112 |
| Cap topology | 94.88% (2,076/2,188) |

Intersection error (2,073 finite residuals, 115 null: 112 inf + 3 zero):
p50=3.60e-6, p95=1.05e-4, p99=2.18e-4, max=6.51e-4. Scale-relative: p50=0.045, p95=1.33, p99=2.75, max=8.20.

## Aggregate

| Metric | Value |
|---|---|
| Total clips | 25,062 |
| Classification rate | 99.55% |
| Oracle rate | 99.9999% (166,659,465/166,659,512) |
| Oracle disagree vertices | 47 / 166M |

## Failure taxonomy

- **Face-only failures**: 0 (across all fixtures)
- **Coord-only failures**: 201 (38 + 2 + 161)
- **Both failures**: 114 (2 + 0 + 112) — all classification disagreements
- All 114 both-failures have null residual (no intersection correspondence)
- Coord-only failures are intersection-point positional drift, not connectivity changes

## Gate criteria

| Criterion | Result |
|---|---|
| Classification > 99% | PASS — 99.55% |
| Oracle > 99.99% | PASS — 99.9999% |
| Zero invalid geometry | PASS |
| Zero unexplained skips | PASS — 108 skips, all classified |
| Topology preserved on disagreements | **FAIL — 0/114** |
| Stratified topology sample | PASS — 3,076 agreeing clips, 100% face-set preservation |
