import numpy as np
import pytest

from chitin.cap_triangulation import triangulate_polygon
from chitin.trace_fixtures import h_shape_calibration_mesh


def test_collinear_boundary_vertices_are_preserved():
    p = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [0, 1]])
    faces = triangulate_polygon(np.arange(8), p)
    assert len(faces) == 6
    assert all(
        np.linalg.det(np.stack([p[b] - p[a], p[c] - p[a]])) > 0 for a, b, c in faces
    )


def test_self_intersecting_boundary_rejected():
    with pytest.raises(ValueError, match="self-intersecting"):
        triangulate_polygon(np.arange(4), np.array([[0, 0], [1, 1], [0, 1], [1, 0]]))


def test_h_shape_source_cap_is_valid():
    v, f = h_shape_calibration_mesh()
    n = len(v) // 2
    p = v[:n, :2].astype(float)
    polygon_area = (
        np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)) / 2
    )
    tris = v[f[np.all(f >= n, axis=1)]].astype(float)
    areas = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])[:, 2] / 2
    assert np.all(areas > 0)
    assert areas.sum() == pytest.approx(polygon_area)
