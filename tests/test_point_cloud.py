# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
import numpy as np

from chitin import Config, extract_from_arrays


def test_sphere_point_cloud(sphere_points):
    r = extract_from_arrays(
        sphere_points, normals=sphere_points, config=Config(concavity=0.5)
    )
    assert len(r.hulls) >= 1
    assert r.source_vertex_count == len(sphere_points)


def test_too_few_points_returns_empty():
    pts = np.random.default_rng(0).standard_normal((50, 3))
    r = extract_from_arrays(pts)
    assert len(r.hulls) == 0
    assert r.source_vertex_count == 50


def test_opacity_filtering():
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((1000, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    opacity = np.ones(1000, dtype=np.float64)
    opacity[:900] = 0.0

    r = extract_from_arrays(pts, opacity=opacity, config=Config(opacity_threshold=0.5))
    assert r.source_vertex_count == 1000


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


def test_all_zero_normals_triggers_estimation(sphere_points):
    zero_normals = np.zeros_like(sphere_points)
    r = extract_from_arrays(
        sphere_points, normals=zero_normals, config=Config(concavity=0.5)
    )
    assert len(r.hulls) >= 1
