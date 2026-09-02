"""Orthogonal topology analysis tests."""

import numpy as np
import trimesh

from chitin.topology import analyze_topology


def _closed_box():
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    return (
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
    )


def _open_mesh():
    """A single triangle — open boundary, not closed."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return verts, faces


def _non_manifold_mesh():
    """Two triangles sharing one edge with a third triangle bridging them —
    creates a non-manifold edge (shared by >2 faces)."""
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 4],
        ],
        dtype=np.int32,
    )
    return verts, faces


def _degenerate_mesh():
    """Triangle with two identical vertices — zero area."""
    verts = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return verts, faces


def _two_component_mesh():
    """Two separate boxes — 2 connected components."""
    box1 = trimesh.creation.box(extents=[1, 1, 1])
    box2 = trimesh.creation.box(
        extents=[1, 1, 1],
        transform=trimesh.transformations.translation_matrix([5, 0, 0]),
    )
    mesh = trimesh.util.concatenate([box1, box2])
    return (
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
    )


class TestClosedBox:
    def test_closed(self):
        v, f = _closed_box()
        topo = analyze_topology(v, f)
        assert topo.closed is True

    def test_two_manifold(self):
        v, f = _closed_box()
        topo = analyze_topology(v, f)
        assert topo.two_manifold is True

    def test_no_boundary_edges(self):
        v, f = _closed_box()
        topo = analyze_topology(v, f)
        assert topo.boundary_edge_count == 0

    def test_no_non_manifold_edges(self):
        v, f = _closed_box()
        topo = analyze_topology(v, f)
        assert topo.non_manifold_edge_count == 0

    def test_no_degenerate_faces(self):
        v, f = _closed_box()
        topo = analyze_topology(v, f)
        assert topo.degenerate_face_count == 0

    def test_single_component(self):
        v, f = _closed_box()
        topo = analyze_topology(v, f)
        assert topo.component_count == 1

    def test_positive_volume(self):
        v, f = _closed_box()
        topo = analyze_topology(v, f)
        assert len(topo.signed_volume_by_component) == 1
        assert abs(topo.signed_volume_by_component[0]) > 0

    def test_consistently_oriented(self):
        v, f = _closed_box()
        topo = analyze_topology(v, f)
        assert topo.consistently_oriented is True

    def test_admits_webgpu(self):
        v, f = _closed_box()
        topo = analyze_topology(v, f)
        assert topo.admits_backend("webgpu") is True

    def test_admits_native(self):
        v, f = _closed_box()
        topo = analyze_topology(v, f)
        assert topo.admits_backend("native") is True


class TestOpenMesh:
    def test_not_closed(self):
        v, f = _open_mesh()
        topo = analyze_topology(v, f)
        assert topo.closed is False

    def test_has_boundary_edges(self):
        v, f = _open_mesh()
        topo = analyze_topology(v, f)
        assert topo.boundary_edge_count == 3

    def test_rejects_webgpu(self):
        v, f = _open_mesh()
        topo = analyze_topology(v, f)
        assert topo.admits_backend("webgpu") is False


class TestNonManifold:
    def test_non_manifold_edges(self):
        v, f = _non_manifold_mesh()
        topo = analyze_topology(v, f)
        assert topo.non_manifold_edge_count > 0

    def test_not_two_manifold(self):
        v, f = _non_manifold_mesh()
        topo = analyze_topology(v, f)
        assert topo.two_manifold is False

    def test_rejects_webgpu(self):
        v, f = _non_manifold_mesh()
        topo = analyze_topology(v, f)
        assert topo.admits_backend("webgpu") is False


class TestDegenerate:
    def test_degenerate_faces(self):
        v, f = _degenerate_mesh()
        topo = analyze_topology(v, f)
        assert topo.degenerate_face_count > 0

    def test_rejects_webgpu(self):
        v, f = _degenerate_mesh()
        topo = analyze_topology(v, f)
        assert topo.admits_backend("webgpu") is False


class TestMultiComponent:
    def test_two_components(self):
        v, f = _two_component_mesh()
        topo = analyze_topology(v, f)
        assert topo.component_count == 2

    def test_two_volumes(self):
        v, f = _two_component_mesh()
        topo = analyze_topology(v, f)
        assert len(topo.signed_volume_by_component) == 2

    def test_still_closed(self):
        v, f = _two_component_mesh()
        topo = analyze_topology(v, f)
        assert topo.closed is True


class TestEmptyMesh:
    def test_empty(self):
        v = np.array([], dtype=np.float32).reshape(0, 3)
        f = np.array([], dtype=np.int32).reshape(0, 3)
        topo = analyze_topology(v, f)
        assert topo.component_count == 0
        assert topo.closed is False


class TestToDict:
    def test_roundtrip(self):
        v, f = _closed_box()
        topo = analyze_topology(v, f)
        d = topo.to_dict()
        assert isinstance(d, dict)
        assert set(d.keys()) == {
            "component_count",
            "boundary_edge_count",
            "non_manifold_edge_count",
            "degenerate_face_count",
            "consistently_oriented",
            "signed_volume_by_component",
            "closed",
            "two_manifold",
        }
