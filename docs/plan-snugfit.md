# Phase 4 design -- snug-fit hull refinement

Expands the phase 4 sketch in `plan-quality.md`. CoACD hulls over-inflate
relative to the input samples; today the only counterweight is
threshold-tuned consolidation. Face plane offsets are continuous, so fit
becomes an optimization instead of a heuristic. scipy only, no new deps,
no .phys format changes.

## Where it slots

A new `stages/snugfit.py` pass over the final hull set, after all
reconciliation (dedup, containment cull, consolidation, occlusion cull)
and before `coverage_report` in both `core.extract_from_arrays` and
`splat.extract_spatial`. Per-hull and independent, so deterministic and
trivially parallel later; ships single-threaded first. Off by default
behind `snug_fit` in config until the numbers earn a flip.

## Parameterization

Per hull: `normals, d0 = outward_face_planes(hull)`. Normals stay fixed
-- only the offsets `d` move, so the hull remains a valid polytope and
the optimization is low-dimensional (faces count, typically tens).

Variables are per-face shrink amounts `s >= 0` with `d = d0 - s`,
bounded `0 <= s_i <= n_i . c - eps` where `c` is the hull centroid:
shrink-only (never grow into neighbors' territory), and every face stays
on the far side of the centroid so the polytope cannot collapse or go
empty.

## Objective

Assigned points `P`: input samples covered by the hull at `d0` (same
AABB prefilter + `point_plane_margins` + tol as `coverage_report`, same
deterministic subsample cap).

The original sketch (and the first implementation) used the CvxNet
LogSumExp smooth indicator with L-BFGS over `s`. That turned out to be
unnecessary: with normals fixed and coverage treated as a hard
constraint, volume is monotone in each face offset, so the minimum-volume
polytope containing `P` is closed-form -- every face independently moves
to the point set's support in its normal direction:

    d_i* = max_p (n_i . p) + tol/2

clamped to the shrink-only / centroid bounds above. Exact, no
iterations, no `beta`/`lambda` tuning. (The L-BFGS variant also stalled
at an equilibrium offset proportional to point density -- the softplus
pull-back force grows with the number of points near a face -- which the
unit test caught: an 8x-inflated box only shrank to ~2x.) The smooth
indicator becomes relevant again only if normals are ever optimized or
coverage is traded off; the reference stays for that future.

Deterministic: pure numpy on the seeded point subsample.

## Rebuilding the hull

Compute the new vertex set with `scipy.spatial.HalfspaceIntersection`
(interior point = centroid, valid by the centroid bound on the offsets),
then `ConvexHull` on those vertices for the new `Hull.vertices`/`indices`.

Refinement must never make output worse than input, and a halfspace
intersection over a near-degenerate face set (many near-parallel CoACD
faces) can drop points even though the offset math says it cannot --
measured on garden: 8246 assigned points lost across 560 rebuilt hulls,
1.05pp coverage regression, over the 0.1pp gate. So acceptance is a hard
per-hull coverage guard: recompute the assigned set against the rebuilt
hull and keep the original unless every assigned point is still covered.
On garden this rejects 81 of 560 candidate rebuilds and the remaining
479 hold coverage exactly (0.9797 == baseline) while dropping slack_p95
17% and total volume 35%. The guard, not the offset math, is what makes
the pass coverage-safe.

## Acceptance gate

Phase 1 metrics are the gate, on garden + at least one prop scan:

- `covered_fraction` within 0.1pp of unrefined
- `slack_p50` / `slack_p95` drop measurably (this is the point: today
  garden p95 is ~0.021 against tolerance 0.0024, ~9x slack)
- total hull volume reduction reported in `build-plan.json`
  (`snugfit_volume_ratio`), eyeballed in the walktest viewer
- byte-determinism holds across two builds (`determinism_repro.py`)

## Risks

- Runtime: ~800 hulls x one support computation plus a halfspace
  intersection rebuild -- expect seconds, not minutes.
- Over-shrink on sparse hulls (few assigned points): require a minimum
  assigned-point count (reuse the 100-point floor used elsewhere) or
  skip.
- Interaction with consolidation: refine after, so consolidation sees
  the original conservative hulls; revisit ordering if slack numbers
  suggest otherwise.

## Tests

- Unit: cube hull inflated 2x around a dense unit-cube point sample
  shrinks back to ~unit cube (volume ratio < 1.1, all points covered).
- Degenerate: hull with all points on one face; near-empty assignment.
- Pipeline: golden prop scan with `snug_fit` on -- coverage held,
  slack_p95 reduced, deterministic across two runs.

## References

- Deng, Genova, Yazdani, Bouaziz, Hinton, Tagliasacchi. "CvxNet:
  Learnable Convex Decomposition." CVPR 2020. (LogSumExp smooth convex
  indicator)
- `docs/plan-quality.md` phase 4 sketch; phase 1 coverage machinery in
  `src/chitin/verify/coverage.py`.
