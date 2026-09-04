# Oracle On-Plane Contract

**Issue:** #123
**Status:** Implemented

## Problem

C++ CoACD's `plane.Side()` returns -1/0/+1, where 0 means the vertex lies on the
clipping plane. The Python f32 classifier computes a nonzero machine-epsilon dot
product for these vertices and returns +1 or -1. This convention difference caused
1,646 false oracle disagreements in the Policy 0.2.0 holdout.

## CoACD's on-plane behavior

1. `plane.Side(vertex)` returns 0 when the vertex is on the plane (within
   floating-point equality of the plane equation).
2. For triangles where **all three** vertices are `Side()==0`, `CutSide()` picks a
   nonzero side for the whole triangle.
3. For **mixed** triangles (some zero, some nonzero), on-plane vertices keep
   `side=0` in the per-vertex record. The mesh split routes them with the nonzero
   majority.
4. The trace records **post-CutSide** values, so `oracle_sides` can contain 0 only
   from mixed triangles.

## Contract

**When `oracle_side == 0`, any f32 or WGSL classification is valid.**

On-plane vertices are genuinely degenerate — neither halfspace is geometrically
correct. The C++ convention (return 0) and the f32 convention (return +1 or -1 from
a machine-epsilon dot product) are both internally consistent. The grader must not
penalize this convention difference.

### Grader rules

| oracle_side | f32_side | Verdict |
|-------------|----------|---------|
| 0 | +1 | **Excused** |
| 0 | -1 | **Excused** |
| 0 | 0 | Agree |
| +1 | +1 | Agree |
| +1 | -1 | **Genuine disagree** |
| -1 | -1 | Agree |
| -1 | +1 | **Genuine disagree** |

## Distance guard

Not all `oracle_side=0` labels are plausible. A vertex recorded as on-plane but
whose f64 signed distance is large (e.g. 1.0) indicates trace corruption, not a
legitimate tie-breaking difference. The grader applies a scale-relative f32 error
bound:

    on_plane_bound = 8 * eps_f32 * max(|v - p|, 1.0)

where `eps_f32 ≈ 1.19e-7` and `|v - p|` is the Euclidean distance from the vertex
to the plane point. The factor 8 is a safety margin over the Higham bound for a 3D
dot product (`2n - 1 = 5` for `n = 3`).

An `oracle_side=0` vertex is excused only when `|dot_f64| ≤ on_plane_bound`.
Otherwise it counts as a genuine disagreement.

### Metrics

- `agreement_rate`: `(num_agree + on_plane_excused) / num_vertices` — the official
  oracle metric; excuses on-plane convention differences.
- `strict_agreement_rate`: `num_agree / num_vertices` — for audit; counts on-plane
  as disagreements.
- `num_disagree`: genuine disagreements only (both nonzero, different signs).

## WGSL portability

The same contract applies to the WGSL classifier: on-plane vertices need not agree
with either the Python f32 classifier or the C++ oracle. All three implementations
are free to break the tie differently.
