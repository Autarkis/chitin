import numpy as np

from chitin.plan import BuildPlan
from chitin.report import build_compilation_report
from chitin.resolve import ResolvedConfig
from chitin.stages.decompose import decompose_and_build
from chitin.stages.flatness import is_flat_mesh, make_planar_box
from chitin.verify.convex import outward_face_planes, points_inside


def _resolved_config(**overrides):
    defaults = {
        "concavity": 0.05,
        "opacity_threshold": 0.5,
        "poisson_depth": 6,
        "min_hull_vertices": 4,
        "max_hulls": 2048,
        "opacity_is_logit": False,
        "coacd_preprocess_mode": "auto",
        "coacd_preprocess_resolution": 50,
        "coacd_adaptive_preprocess": True,
        "coacd_deterministic": True,
        "coacd_timeout": 300.0,
        "max_decompose_vertices": 200_000,
        "lod_concavities": None,
        "splat_scale_is_log": True,
        "splat_surface_ratio": 0.2,
        "spatial_split_threshold": 50_000,
        "poisson_density_quantile": 0.1,
        "surface_proximity_filter": 0.0,
        "thin_shell": False,
        "thin_shell_thickness": 0.0,
        "flatness_threshold": 0.9,
        "auto_environment": True,
        "force_environment": True,
        "seam_repair": True,
        "snug_fit": False,
        "use_spatial_split": False,
        "use_seam_repair": False,
        "pipeline_path": "mesh",
    }
    defaults.update(overrides)
    return ResolvedConfig(**defaults)


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
    _flat, normal = is_flat_mesh(vertices, faces, threshold=0.9)
    hull = make_planar_box(vertices, normal)
    extents = hull.vertices.max(axis=0) - hull.vertices.min(axis=0)
    assert extents.min() > 0.0


def test_report_counts_planar_substitute_hulls_for_walkable_floor():
    vertices, faces = _grid_mesh(n=17)
    plan = BuildPlan(input_kind="mesh", collider_kind="static")
    plan.detected["is_environment"] = True
    config = _resolved_config()
    result = decompose_and_build(
        vertices, faces, len(vertices), len(vertices), config, _plan=plan
    )
    report = build_compilation_report(result)
    assert report.processing["fallbacks"]["planar_substitute_hulls"] > 0


def test_report_planar_substitute_hulls_zero_without_substitution():
    vertices, faces = _grid_mesh(n=4)
    plan = BuildPlan(input_kind="mesh", collider_kind="static")
    config = _resolved_config(force_environment=False, auto_environment=False)
    result = decompose_and_build(
        vertices, faces, len(vertices), len(vertices), config, _plan=plan
    )
    report = build_compilation_report(result)
    assert report.processing["fallbacks"]["planar_substitute_hulls"] == 0
    assert report.processing["fallbacks"]["planar_substitute_hulls"] is not None
