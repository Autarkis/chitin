import numpy as np

from chitin.stages.flatness import is_flat_mesh, make_planar_box
from chitin.verify.convex import outward_face_planes, points_inside


def _grid_mesh(n=10, height_fn=None):
    """Triangulated (n x n) grid over [-1, 1]^2; z from height_fn or 0."""
    xs = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(xs, xs)
    zz = height_fn(xx, yy) if height_fn else np.zeros_like(xx)
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    faces = []
    for r in range(n - 1):
        for c in range(n - 1):
            i = r * n + c
            faces.append([i, i + 1, i + n])
            faces.append([i + 1, i + n + 1, i + n])
    return vertices, np.asarray(faces, dtype=np.int32)


def test_flat_plane_detected_at_default_threshold():
    vertices, faces = _grid_mesh()
    flat, normal = is_flat_mesh(vertices, faces, threshold=0.9)
    assert flat
    np.testing.assert_allclose(np.abs(normal), [0.0, 0.0, 1.0], atol=1e-9)


def test_flat_plane_detected_at_threshold_one():
    # A perfectly planar mesh has dominant ratio exactly 1.0.
    vertices, faces = _grid_mesh()
    flat, _ = is_flat_mesh(vertices, faces, threshold=1.0)
    assert flat


def test_paraboloid_not_flat_at_default_threshold():
    vertices, faces = _grid_mesh(height_fn=lambda x, y: 2.0 * (x**2 + y**2))
    flat, normal = is_flat_mesh(vertices, faces, threshold=0.9)
    assert not flat
    assert normal is None


def test_paraboloid_flat_at_permissive_threshold():
    # The same curved mesh passes when the threshold is loosened, which is
    # the knob --flatness-threshold exposes.
    vertices, faces = _grid_mesh(height_fn=lambda x, y: 2.0 * (x**2 + y**2))
    flat, _ = is_flat_mesh(vertices, faces, threshold=0.3)
    assert flat


def test_gentle_slope_flat_at_default_threshold():
    vertices, faces = _grid_mesh(height_fn=lambda x, y: 0.05 * x)
    flat, _ = is_flat_mesh(vertices, faces, threshold=0.9)
    assert flat


def test_make_planar_box_contains_all_vertices():
    rng = np.random.default_rng(3)
    vertices, faces = _grid_mesh(
        height_fn=lambda x, y: 0.02 * rng.standard_normal(x.shape)
    )
    flat, normal = is_flat_mesh(vertices, faces, threshold=0.9)
    assert flat

    hull = make_planar_box(vertices, normal)
    normals, d = outward_face_planes(hull)
    inside = points_inside(normals, d, vertices, tol=1e-6)
    assert inside.all()


def test_make_planar_box_has_minimum_thickness():
    vertices, faces = _grid_mesh()  # exactly planar: zero natural thickness
    flat, normal = is_flat_mesh(vertices, faces, threshold=0.9)
    hull = make_planar_box(vertices, normal)
    extents = hull.vertices.max(axis=0) - hull.vertices.min(axis=0)
    assert extents.min() > 0.0
