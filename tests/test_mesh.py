import numpy as np

from chitin import Config, extract_from_mesh


def test_box_produces_hulls(box_mesh):
    verts, faces = box_mesh
    r = extract_from_mesh(verts, faces, config=Config(concavity=0.5))
    assert len(r.hulls) >= 1
    assert r.source_vertex_count == len(verts)
    assert r.mesh_vertex_count == len(verts)
    assert r.bones is None


def test_hull_vertices_are_float32(box_mesh):
    verts, faces = box_mesh
    r = extract_from_mesh(verts, faces, config=Config(concavity=0.5))
    for hull in r.hulls:
        assert hull.vertices.dtype == np.float32
        assert hull.indices.dtype == np.uint32
        assert hull.vertices.ndim == 2
        assert hull.vertices.shape[1] == 3


def test_hull_vertices_within_input_bounds(box_mesh):
    verts, faces = box_mesh
    r = extract_from_mesh(verts, faces, config=Config(concavity=0.5))
    input_min = verts.min(axis=0) - 0.1
    input_max = verts.max(axis=0) + 0.1
    for hull in r.hulls:
        assert np.all(hull.vertices >= input_min)
        assert np.all(hull.vertices <= input_max)


def test_empty_mesh_returns_empty():
    verts = np.zeros((0, 3), dtype=np.float64)
    faces = np.zeros((0, 3), dtype=np.int32)
    r = extract_from_mesh(verts, faces)
    assert len(r.hulls) == 0


def test_degenerate_single_triangle():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    r = extract_from_mesh(verts, faces)
    assert len(r.hulls) == 0


def test_max_hulls_respected(box_mesh):
    verts, faces = box_mesh
    r = extract_from_mesh(verts, faces, config=Config(concavity=0.01, max_hulls=1))
    assert len(r.hulls) <= 1
