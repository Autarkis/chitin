# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
# (new pipeline stage, not a script; design in docs/plan-snugfit.md)
"""Snug-fit hull refinement (phase 4): shrink face plane offsets onto the
covered input samples. Design: docs/plan-snugfit.md."""

from __future__ import annotations

import numpy as np

from chitin.result import Hull
from chitin.stages.occlusion import _covered_indices
from chitin.verify.convex import outward_face_planes
from chitin.verify.coverage import MAX_COVERAGE_SAMPLES

MIN_ASSIGNED_POINTS = 100


def refine_hulls(
    hulls: list, points: np.ndarray | None, tol_fraction: float = 0.001
) -> tuple[list, dict]:
    """Tighten each hull onto the input points it covers.

    Per hull: fix the outward face normals and move each face plane to
    the assigned points' support in its normal direction -- the exact
    minimum-volume fixed-normal polytope containing the points, so no
    iterative optimization is needed -- then rebuild the polytope via
    halfspace intersection. Shrink-only, so hulls never grow.

    Refinement is coverage-safe by construction: a rebuilt hull is
    accepted only if it still covers every input point the original
    covered (the halfspace-intersection rebuild can drop points on
    near-degenerate face sets even though the offset math would not). On
    any rejection or geometry failure the original hull is kept, so the
    pass can only hold or improve coverage. Hulls covering fewer than
    MIN_ASSIGNED_POINTS input samples are skipped.

    Deterministic. Returns (hulls, stats) where stats is JSON-ready for
    build-plan ``detected``. No-op without scipy.
    """
    if not hulls or points is None or len(points) == 0:
        return hulls, {}
    try:
        from scipy.spatial import QhullError  # noqa: F401
    except ImportError:
        return hulls, {}

    points = np.asarray(points, dtype=np.float64)
    if len(points) > MAX_COVERAGE_SAMPLES:
        rng = np.random.default_rng(0)
        choice = rng.choice(len(points), size=MAX_COVERAGE_SAMPLES, replace=False)
        points = points[np.sort(choice)]

    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    tol = tol_fraction * diagonal

    out = []
    refined = 0
    rejected = 0
    vol_before = 0.0
    vol_after = 0.0
    for hull in hulls:
        idx = _covered_indices(hull, points, tol)
        if len(idx) < MIN_ASSIGNED_POINTS:
            out.append(hull)
            continue
        result = _refine_one(hull, points[idx], tol)
        if result is None:
            out.append(hull)
            continue
        new_hull, before, after = result
        # Coverage guard: reject any rebuild that drops an assigned point.
        if len(_covered_indices(new_hull, points[idx], tol)) < len(idx):
            rejected += 1
            out.append(hull)
            continue
        out.append(new_hull)
        refined += 1
        vol_before += before
        vol_after += after

    stats: dict = {
        "snugfit_refined": refined,
        "snugfit_rejected": rejected,
        "snugfit_skipped": len(hulls) - refined - rejected,
    }
    if vol_before > 0:
        stats["snugfit_volume_ratio"] = round(vol_after / vol_before, 4)
    return out, stats


def _unique_planes(normals: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge coplanar triangles into one halfspace, so a quad face
    contributes one plane instead of two identical ones."""
    key = np.round(np.column_stack([normals, d]), 6)
    _, first = np.unique(key, axis=0, return_index=True)
    return normals[first], d[first]


def _refine_one(hull, pts: np.ndarray, tol: float):
    """Shrink one hull onto its assigned points. Returns (new_hull,
    volume_before, volume_after) or None to keep the original.

    With normals fixed and coverage a hard constraint, the minimum-volume
    polytope is closed-form: volume is monotone in each face offset, so
    every face independently moves to the point set's support in its
    normal direction, max_p(n_i . p), plus half the coverage tolerance.
    """
    from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

    normals, d0 = _unique_planes(*outward_face_planes(hull))

    verts = hull.vertices.astype(np.float64)
    centroid = verts.mean(axis=0)
    # Shrink-only, and every plane stays on the far side of the centroid
    # so the polytope cannot collapse or go empty.
    headroom = d0 - normals @ centroid
    if np.any(headroom <= tol):
        return None

    support = (pts @ normals.T).max(axis=0)
    shrink = np.clip(d0 - support - 0.5 * tol, 0.0, 0.95 * headroom)
    d = d0 - shrink

    try:
        intersection = HalfspaceIntersection(
            np.hstack([normals, -d[:, None]]), centroid
        )
        new_ch = ConvexHull(intersection.intersections)
        volume_before = float(ConvexHull(verts).volume)
    except QhullError:
        return None
    if new_ch.volume <= 0 or new_ch.volume > volume_before:
        return None

    vmap = np.full(len(new_ch.points), -1, dtype=np.int64)
    vmap[new_ch.vertices] = np.arange(len(new_ch.vertices))
    new_hull = Hull(
        vertices=new_ch.points[new_ch.vertices].astype(np.float32),
        indices=vmap[new_ch.simplices].ravel().astype(np.uint32),
        bone_name=hull.bone_name,
        bone_index=hull.bone_index,
    )
    return new_hull, volume_before, float(new_ch.volume)
