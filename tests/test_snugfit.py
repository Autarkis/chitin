# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
# (pytest module for the new snugfit stage)
import numpy as np
import pytest

from chitin.result import Hull

pytest.importorskip("scipy")

from chitin.stages.snugfit import refine_hulls  # noqa: E402
from chitin.verify.convex import outward_face_planes, point_plane_margins  # noqa: E402


def _box_hull(center=(0.0, 0.0, 0.0), half=(1.0, 1.0, 1.0)):
    c = np.array(center, dtype=np.float32)
    h = np.array(half, dtype=np.float32)
    signs = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
        dtype=np.float32,
    )
    verts = c + h * signs
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 5, 7],
            [4, 7, 6],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 3, 7],
            [1, 7, 5],
        ],
        dtype=np.uint32,
    )
    return Hull(vertices=verts, indices=faces.ravel())


def _cube_grid(half=0.5, n=12):
    axis = np.linspace(-half, half, n)
    return np.stack(np.meshgrid(axis, axis, axis), axis=-1).reshape(-1, 3)


def _hull_volume(hull):
    from scipy.spatial import ConvexHull

    return ConvexHull(hull.vertices.astype(np.float64)).volume


def _all_covered(hull, points, tol):
    normals, d = outward_face_planes(hull)
    return bool(np.all(point_plane_margins(normals, d, points) >= -tol))


def test_inflated_box_shrinks_onto_points():
    # 2x-inflated box around a dense unit-cube sample: volume 8 -> ~1.
    hull = _box_hull(half=(1.0, 1.0, 1.0))
    points = _cube_grid(half=0.5)

    refined, stats = refine_hulls([hull], points)

    assert stats["snugfit_refined"] == 1
    assert stats["snugfit_volume_ratio"] < 0.3
    assert _hull_volume(refined[0]) < 1.5
    # No covered point may be pushed out beyond the coverage tolerance.
    tol = 0.001 * float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    assert _all_covered(refined[0], points, tol)


def test_tight_hull_barely_changes():
    # Hull already snug on the points: volume must not collapse.
    hull = _box_hull(half=(0.5, 0.5, 0.5))
    points = _cube_grid(half=0.5)

    refined, stats = refine_hulls([hull], points)

    assert stats["snugfit_refined"] == 1
    assert _hull_volume(refined[0]) > 0.8 * _hull_volume(hull)


def test_sparse_assignment_skipped():
    # Fewer than MIN_ASSIGNED_POINTS covered samples: hull kept verbatim.
    hull = _box_hull()
    points = _cube_grid(half=0.4, n=4)  # 64 points < 100

    refined, stats = refine_hulls([hull], points)

    assert stats["snugfit_refined"] == 0
    assert refined[0] is hull


def test_uncovered_hull_untouched():
    # Hull far away from every point: nothing assigned, kept verbatim.
    hull = _box_hull(center=(50.0, 0.0, 0.0))
    near = _box_hull(half=(1.0, 1.0, 1.0))
    points = _cube_grid(half=0.5)

    refined, stats = refine_hulls([near, hull], points)

    assert stats["snugfit_refined"] == 1
    assert refined[1] is hull


def test_refinement_is_deterministic():
    hull = _box_hull(half=(1.0, 1.0, 1.0))
    points = _cube_grid(half=0.5)

    a, _ = refine_hulls([hull], points)
    b, _ = refine_hulls([hull], points)

    assert np.array_equal(a[0].vertices, b[0].vertices)
    assert np.array_equal(a[0].indices, b[0].indices)
