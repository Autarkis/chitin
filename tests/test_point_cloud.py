# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
import numpy as np
import pytest

from chitin import Config, extract_from_arrays

try:
    import open3d  # noqa: F401

    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False
requires_open3d = pytest.mark.skipif(not _HAS_OPEN3D, reason="requires chitin[splat]")


@requires_open3d
def test_sphere_point_cloud(sphere_points):
    r = extract_from_arrays(
        sphere_points, normals=sphere_points, config=Config(concavity=0.5)
    )
    assert len(r.hulls) >= 1
    assert r.source_vertex_count == len(sphere_points)


@requires_open3d
def test_too_few_points_returns_empty():
    pts = np.random.default_rng(0).standard_normal((50, 3))
    r = extract_from_arrays(pts)
    assert len(r.hulls) == 0
    assert r.source_vertex_count == 50


@requires_open3d
def test_opacity_filtering():
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((1000, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    opacity = np.ones(1000, dtype=np.float64)
    opacity[:900] = 0.0

    r = extract_from_arrays(pts, opacity=opacity, config=Config(opacity_threshold=0.5))
    assert r.source_vertex_count == 1000


@requires_open3d
def test_logit_opacity():
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((500, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    logits = np.full(500, 5.0, dtype=np.float64)

    r = extract_from_arrays(
        pts,
        opacity=logits,
        normals=pts,
        config=Config(opacity_is_logit=True, concavity=0.5),
    )
    assert len(r.hulls) >= 1


@requires_open3d
def test_all_zero_normals_triggers_estimation(sphere_points):
    zero_normals = np.zeros_like(sphere_points)
    r = extract_from_arrays(
        sphere_points, normals=zero_normals, config=Config(concavity=0.5)
    )
    assert len(r.hulls) >= 1


def test_normals_from_covariance_identity_quat():
    from chitin.stages.splat import normals_from_covariance

    scales = np.array(
        [
            [np.log(5.0), np.log(5.0), np.log(0.1)],
            [np.log(0.1), np.log(5.0), np.log(5.0)],
            [np.log(5.0), np.log(0.1), np.log(5.0)],
        ]
    )
    rots = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float64)

    normals = normals_from_covariance(scales, rots, log_scale=True)

    np.testing.assert_allclose(np.abs(normals[0]), [0, 0, 1], atol=1e-10)
    np.testing.assert_allclose(np.abs(normals[1]), [1, 0, 0], atol=1e-10)
    np.testing.assert_allclose(np.abs(normals[2]), [0, 1, 0], atol=1e-10)


def test_normals_from_covariance_unit_length():
    from chitin.stages.splat import normals_from_covariance

    rng = np.random.default_rng(99)
    scales = rng.standard_normal((200, 3))
    rots = rng.standard_normal((200, 4))

    normals = normals_from_covariance(scales, rots, log_scale=True)
    lengths = np.linalg.norm(normals, axis=1)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-10)


def test_normals_from_covariance_linear_scale():
    from chitin.stages.splat import normals_from_covariance

    scales = np.array([[5.0, 5.0, 0.1]], dtype=np.float64)
    rots = np.array([[1, 0, 0, 0]], dtype=np.float64)

    normals = normals_from_covariance(scales, rots, log_scale=False)
    np.testing.assert_allclose(np.abs(normals[0]), [0, 0, 1], atol=1e-10)


def test_inflate_splat_points_count():
    from chitin.stages.splat import inflate_splat_points

    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    scales = np.array([[np.log(1.0), np.log(1.0), np.log(0.1)]] * 2, dtype=np.float64)
    rots = np.array([[1, 0, 0, 0]] * 2, dtype=np.float64)

    inflated = inflate_splat_points(positions, scales, rots, surface_ratio=0.2)
    assert len(inflated) == 10  # 2 originals * 5 (center + 4 disk samples)
    np.testing.assert_array_equal(inflated[:2], positions)


def test_octree_partition_small_set():
    from chitin.stages.splat import octree_partition

    rng = np.random.default_rng(7)
    positions = rng.uniform(-10, 10, (100, 3))

    cells = octree_partition(positions, max_points=200)
    assert len(cells) == 1
    assert len(cells[0].indices) == 100


def test_octree_partition_splits():
    from chitin.stages.splat import octree_partition

    rng = np.random.default_rng(7)
    positions = rng.uniform(-10, 10, (1000, 3))

    cells = octree_partition(positions, max_points=200)
    assert len(cells) > 1

    all_indices = np.concatenate([c.indices for c in cells])
    assert len(all_indices) == 1000
    assert len(np.unique(all_indices)) == 1000


def test_octree_partition_bounds_cover_points():
    from chitin.stages.splat import octree_partition

    rng = np.random.default_rng(7)
    positions = rng.uniform(-5, 5, (500, 3))

    cells = octree_partition(positions, max_points=100)
    for cell in cells:
        pts = positions[cell.indices]
        assert np.all(pts >= cell.bounds_min - 1e-10)
        assert np.all(pts <= cell.bounds_max + 1e-10)


def test_octree_partition_max_depth():
    from chitin.stages.splat import octree_partition

    positions = np.zeros((100, 3), dtype=np.float64)
    cells = octree_partition(positions, max_points=10, max_depth=2)
    assert all(len(c.indices) <= 100 for c in cells)


def test_auto_poisson_depth():
    from chitin.resolve import _auto_poisson_depth

    assert _auto_poisson_depth(0) == 4
    assert _auto_poisson_depth(1000) == 4
    assert _auto_poisson_depth(50_000) == 5
    assert _auto_poisson_depth(100_000) == 5
    assert _auto_poisson_depth(500_000) == 6
    assert _auto_poisson_depth(1_000_000) == 6
    assert _auto_poisson_depth(10_000_000) == 7


def test_aabb_iou_identical():
    from chitin.stages.decompose import aabb_iou
    from chitin.result import Hull

    verts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    idx = np.array([0, 1, 0], dtype=np.uint32)
    h = Hull(vertices=verts, indices=idx)
    assert abs(aabb_iou(h, h) - 1.0) < 1e-10


def test_aabb_iou_no_overlap():
    from chitin.stages.decompose import aabb_iou
    from chitin.result import Hull

    idx = np.array([0, 1, 0], dtype=np.uint32)
    a = Hull(vertices=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32), indices=idx)
    b = Hull(vertices=np.array([[5, 5, 5], [6, 6, 6]], dtype=np.float32), indices=idx)
    assert aabb_iou(a, b) == 0.0


def test_aabb_iou_partial():
    from chitin.stages.decompose import aabb_iou
    from chitin.result import Hull

    idx = np.array([0, 1, 0], dtype=np.uint32)
    a = Hull(vertices=np.array([[0, 0, 0], [2, 2, 2]], dtype=np.float32), indices=idx)
    b = Hull(vertices=np.array([[1, 1, 1], [3, 3, 3]], dtype=np.float32), indices=idx)
    iou = aabb_iou(a, b)
    assert 0.0 < iou < 1.0


def test_dedup_removes_duplicate():
    from chitin.stages.decompose import dedup_overlapping_hulls
    from chitin.result import Hull

    idx = np.array([0, 1, 0], dtype=np.uint32)
    a = Hull(vertices=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32), indices=idx)
    b = Hull(
        vertices=np.array([[0.05, 0.05, 0.05], [0.95, 0.95, 0.95]], dtype=np.float32),
        indices=idx,
    )
    result = dedup_overlapping_hulls([a, b], iou_threshold=0.5)
    assert len(result) == 1


def test_dedup_keeps_distinct():
    from chitin.stages.decompose import dedup_overlapping_hulls
    from chitin.result import Hull

    idx = np.array([0, 1, 0], dtype=np.uint32)
    a = Hull(vertices=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32), indices=idx)
    b = Hull(vertices=np.array([[5, 5, 5], [6, 6, 6]], dtype=np.float32), indices=idx)
    result = dedup_overlapping_hulls([a, b], iou_threshold=0.5)
    assert len(result) == 2


def _sphere_with_covariance(n, rng=None):
    rng = rng or np.random.default_rng(42)
    pts = rng.standard_normal((n, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    scales = np.full((n, 3), [np.log(1.0), np.log(1.0), np.log(0.1)], dtype=np.float64)
    rots = _orient_quats_to_sphere(pts)
    return pts, scales, rots


def _orient_quats_to_sphere(pts):

    n = len(pts)
    normals = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    z_axis = np.array([0.0, 0.0, 1.0])
    rots = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        v = np.cross(z_axis, normals[i])
        c = np.dot(z_axis, normals[i])
        if np.linalg.norm(v) < 1e-10:
            rots[i] = [1, 0, 0, 0] if c > 0 else [0, 1, 0, 0]
        else:
            rots[i, 0] = 1.0 + c
            rots[i, 1:] = v
            rots[i] /= np.linalg.norm(rots[i])
    return rots


def test_covariance_normals_match_sphere_surface():
    from chitin.stages.splat import normals_from_covariance

    pts, scales, rots = _sphere_with_covariance(200)
    derived = normals_from_covariance(scales, rots, log_scale=True)
    expected = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    dots = np.abs(np.sum(derived * expected, axis=1))
    assert np.mean(dots) > 0.99


@requires_open3d
def test_covariance_pipeline_end_to_end():
    pts, scales, rots = _sphere_with_covariance(500)
    r = extract_from_arrays(pts, scales=scales, rots=rots, config=Config(concavity=0.5))
    assert len(r.hulls) >= 1
    assert r.build_plan.detected.get("covariance_normals") is True


@requires_open3d
def test_covariance_with_opacity_filtering():
    pts, scales, rots = _sphere_with_covariance(1000)
    opacity = np.ones(1000, dtype=np.float64)
    opacity[:500] = 0.0

    r = extract_from_arrays(
        pts,
        opacity=opacity,
        scales=scales,
        rots=rots,
        config=Config(opacity_threshold=0.5, concavity=0.5),
    )
    assert r.source_vertex_count == 1000
    assert r.build_plan.detected.get("filtered_vertices") == 500


@requires_open3d
def test_covariance_with_lod():
    pts, scales, rots = _sphere_with_covariance(500)
    r = extract_from_arrays(
        pts,
        scales=scales,
        rots=rots,
        config=Config(concavity=0.05, lod_concavities=[0.3, 0.5]),
    )
    assert r.lod_tiers is not None
    assert len(r.lod_tiers) == 2
    assert r.lod_tiers[0].concavity == 0.3
    assert r.lod_tiers[1].concavity == 0.5


@requires_open3d
def test_spatial_split_triggered():
    rng = np.random.default_rng(42)
    pts, scales, rots = _sphere_with_covariance(2000, rng=rng)
    pts = pts * 5.0

    r = extract_from_arrays(
        pts,
        scales=scales,
        rots=rots,
        config=Config(concavity=0.5, spatial_split_threshold=500),
    )
    assert len(r.hulls) >= 1
    assert r.build_plan.detected.get("cell_count", 0) > 1
    assert "reconciled_hulls" in r.build_plan.detected


def test_proximity_filter_removes_distant_vertices():
    from chitin.stages.filter import proximity_filter_mesh

    input_pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    mesh_verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 10, 10]], dtype=np.float64
    )
    mesh_tris = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)

    filtered_verts, filtered_tris = proximity_filter_mesh(
        mesh_verts, mesh_tris, input_pts, max_distance_ratio=3.0
    )
    assert len(filtered_verts) == 3
    assert len(filtered_tris) == 1


def test_extrude_thin_shell_doubles_geometry():
    from chitin.stages.filter import extrude_thin_shell

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    ext_verts, ext_faces = extrude_thin_shell(verts, faces, thickness=0.1)
    assert len(ext_verts) == 8
    assert len(ext_faces) > 4


@requires_open3d
def test_density_quantile_config():
    pts, scales, rots = _sphere_with_covariance(500)
    r_default = extract_from_arrays(
        pts, scales=scales, rots=rots, config=Config(concavity=0.5)
    )
    r_tight = extract_from_arrays(
        pts,
        scales=scales,
        rots=rots,
        config=Config(concavity=0.5, poisson_density_quantile=0.4),
    )
    assert r_default.hulls is not None
    assert r_tight.hulls is not None


@requires_open3d
def test_thin_shell_produces_hulls():
    pts, scales, rots = _sphere_with_covariance(500)
    r = extract_from_arrays(
        pts,
        scales=scales,
        rots=rots,
        config=Config(concavity=0.5, thin_shell=True),
    )
    assert len(r.hulls) >= 1
