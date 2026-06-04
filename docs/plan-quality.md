# Chitin quality feedback plan

The goal is to close the loop. The pipeline makes every quality-relevant decision through unmeasured heuristics -- roughly 30 hardcoded thresholds across the stages -- and ships hulls with no integrated signal for whether the output is good. `verify/probe.py` and `verify/sweep.py` exist but are post-hoc CLI commands; nothing inside extraction measures coverage, tightness, or redundancy.

The current smell, from the garden example bundle (`examples/garden-colliders/`):

- Seam repair moves probe coverage from 72.9% to 76.1%. We only know because someone ran `chitin probe` by hand and pasted the number into a commit message (82a6b8a).
- The consolidation pass drops hull count from 278 to 140. Half the pre-consolidation output is redundant, and nothing in the pipeline reports that.
- Build-plan padding spread is 0.0121 to 0.0428 (3.5x) across cells -- per-cell padding (2453f6d) was added because the global heuristic was provably wrong, found by manual inspection, not by a metric.
- The boundary/reconciliation logic has needed repeated bug fixes (2453f6d dedup ordering, bde64f0 centroid ownership, 82a6b8a seams). Chronic churn in one area with no regression metric is the signature of a missing feedback signal.

Separately, normal sign is unvalidated in three places, and Poisson is sensitive to flipped normals:

- `stages/splat.py` `normals_from_covariance()` -- the min-scale eigenvector has arbitrary sign, never oriented.
- `stages/splat.py` inflation tiles normals 5x verbatim.
- `stages/flatness.py` planar-box dominant eigenvector sign is unspecified.

The estimated-normals path already calls Open3D `orient_normals_consistent_tangent_plane` (Hoppe et al. 1992 lineage); the splat covariance path skips orientation entirely.

## Phase 0 -- instrument what already exists

Record per-stage deltas in `BuildPlan.detected`: hulls in/out of dedup, containment cull, consolidation, and seam repair; decimation ratio when `max_decompose_vertices` triggers; count of cells silently dropped in `_process_single_cell` (currently three `return None` paths with no trace). No behavior change, pure bookkeeping.

Success: garden bundle's `build-plan.json` explains where every hull came from and went.

## Phase 1 -- integrated coverage metric

The input points are the ground truth and we already hold them during extraction. After reconciliation, classify every input sample as covered/uncovered using the existing plane machinery (`_outward_face_planes` + `_per_vertex_inside` in `stages/decompose.py`), plus a slack distance for covered points (how far inside the nearest hull). Report:

- global covered fraction
- per-octree-cell covered fraction, and the worst decile called out explicitly -- means hide holes; a scene at 95% global coverage with one cell at 40% is a fall-through bug
- slack histogram (tightness)

All of it lands in `build-plan.json`. The `probe` raycast tool stays as the independent post-hoc check.

Success: coverage block in the garden bundle; a CI test asserting golden scenes stay above a floor; phases 2-4 are judged by this metric, not by eyeball.

## Phase 2 -- normal hygiene on the splat path

Orient covariance normals before Poisson. Two candidate mechanisms, both already in our dependency tree or trivial:

1. Run the same `orient_normals_consistent_tangent_plane` the estimated path uses.
2. Visibility-based orientation: points visible from an exterior viewpoint should have normals facing it. Open3D ships `hidden_point_removal` (Katz, Tal, Basri, "Direct Visibility of Point Sets", SIGGRAPH 2007); a pass over a few dozen fibonacci-sphere viewpoints votes a sign per point. More robust than MST propagation on thin walls.

Also fix the planar-box eigenvector sign in `stages/flatness.py` (orient against cell centroid).

Success: phase 1 coverage and slack improve (or hold) on garden; add a synthetic two-sided-wall fixture that currently produces flipped normals.

## Phase 3 -- union-occlusion hull culling

`cull_contained_hulls` only removes a hull when a single other hull contains it. Scan-derived geometry produces junk hulls buried inside the union of several hulls -- never collidable, not contained by any one neighbor. The consolidation numbers (278 to 140) say this redundancy is large.

Approach: sample each hull's surface, run `hidden_point_removal` from fibonacci-sphere viewpoints over the full hull set, cull hulls with zero exterior-visible samples. Skip when `is_environment` (interiors are walkable there) -- the flag already exists in `analyze.py`.

Success: hull count drops on prop scans with zero phase 1 coverage regression; the existing consolidation heuristics (five coupled thresholds in `consolidate_near_contained_hulls`) get less load-bearing.

## Phase 4 -- snug-fit refinement (stretch)

CoACD hulls over-inflate relative to the input samples, and today the only counterweight is threshold-tuned consolidation. Hull face plane offsets are continuous parameters, so fit can be an optimization instead of a heuristic: a smooth point-in-convex indicator via LogSumExp over plane slacks (the relaxation used by CvxNet, Deng et al., CVPR 2020), minimizing hull volume plus a penalty on uncovered input samples. scipy L-BFGS on plane offsets `d` only -- normals fixed, so hulls stay valid polytopes. No new dependencies.

This is a real project, not a patch. Gate on phases 0-3 shipping first; phase 1 provides the objective and the regression gate.

## Test debt (alongside each phase)

The evidence pass found zero tests for seam detection/repair, flatness detection, and the environment heuristic -- the three areas with the most commit churn. Each phase above lands with fixtures:

- flat-cell scene (exercises `flatness.py` at thresholds 0/0.5/0.9/1.0)
- two-cell seam scene (exercises `verify/seam.py` and `stages/repair.py`)
- hollow-shell environment scene (exercises auto-detection at the 5% boundary)
- regression tests for the dedup ordering (2453f6d) and AABB reconciliation (bde64f0) fixes, which currently have one trivial test between them

## Non-goals

No new input formats, export targets, or features. No GPU or torch dependency -- phase 4 is deliberately scoped to scipy. No changes to the .phys format.

## References

- Hoppe, DeRose, Duchamp, McDonald, Stuetzle. "Surface Reconstruction from Unorganized Points." SIGGRAPH 1992. (normal orientation propagation)
- Katz, Tal, Basri. "Direct Visibility of Point Sets." SIGGRAPH 2007. (hidden point removal; `open3d.geometry.PointCloud.hidden_point_removal`)
- Deng, Genova, Yazdani, Bouaziz, Hinton, Tagliasacchi. "CvxNet: Learnable Convex Decomposition." CVPR 2020. (smooth convex indicator via LogSumExp over hyperplane slacks)
- Wei, Liu, Zhao, Xu et al. "Approximate Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree Search." SIGGRAPH 2022. (CoACD; Hausdorff-based concavity metric)
