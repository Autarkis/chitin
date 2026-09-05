"""GPU clip stage tests — count/scan/emit pipeline properties."""

import numpy as np
import pytest

from chitin.f32_policy import DEFAULT_POLICY
from chitin.f32_predicates import classify_plane_f32
from chitin.gpu.clip import ClipEmitResult, dispatch_clip_emit_reference
from chitin.gpu.worker import GPUWorker


@pytest.fixture(scope="module")
def worker():
    if not GPUWorker.available():
        pytest.skip("No GPU adapter available")
    with GPUWorker() as w:
        yield w


def _cpu_classify(vertices, plane_point, plane_normal):
    """Run CPU classification, return (signs, world_dots)."""
    result = classify_plane_f32(
        vertices.astype(np.float64),
        plane_point.astype(np.float64),
        plane_normal.astype(np.float64),
        DEFAULT_POLICY,
    )
    return result.signs.astype(np.int32), result.signed_distances.astype(np.float32)


class TestGPUClipEmitReference:
    def test_all_positive(self, worker):
        vertices = np.array(
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        plane_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        signs, dots = _cpu_classify(vertices, plane_point, plane_normal)
        result = dispatch_clip_emit_reference(worker, vertices, faces, signs, dots)

        assert isinstance(result, ClipEmitResult)
        assert result.total_positive_faces == 1
        assert result.total_negative_faces == 0
        assert result.total_intersections == 0
        assert result.total_boundary_edges == 0
        np.testing.assert_array_equal(np.sort(result.positive_faces[0]), [0, 1, 2])
        np.testing.assert_array_equal(result.positive_ancestry, [0])

    def test_all_negative(self, worker):
        vertices = np.array(
            [[0.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, -1.0, 1.0]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        plane_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        signs, dots = _cpu_classify(vertices, plane_point, plane_normal)
        result = dispatch_clip_emit_reference(worker, vertices, faces, signs, dots)

        assert result.total_negative_faces == 1
        assert result.total_positive_faces == 0
        assert result.total_intersections == 0
        np.testing.assert_array_equal(np.sort(result.negative_faces[0]), [0, 1, 2])
        np.testing.assert_array_equal(result.negative_ancestry, [0])

    def test_single_triangle_split(self, worker):
        # v0 above the plane, v1 and v2 below.
        vertices = np.array(
            [[0.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        plane_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        signs, dots = _cpu_classify(vertices, plane_point, plane_normal)
        assert signs[0] == 1
        assert signs[1] == -1
        assert signs[2] == -1

        result = dispatch_clip_emit_reference(worker, vertices, faces, signs, dots)

        assert result.total_positive_faces == 1
        assert result.total_negative_faces == 2
        assert result.total_intersections == 2
        assert result.total_boundary_edges == 1

        # Intersection points lie on the y=0 plane.
        np.testing.assert_allclose(result.intersection_positions[:, 1], 0.0, atol=1e-5)

        # EdgeKeys are valid (v_lo < v_hi) and reference source vertices.
        for v_lo, v_hi in result.intersection_edges:
            assert v_lo < v_hi
            assert v_lo < 3
            assert v_hi < 3

        np.testing.assert_array_equal(result.positive_ancestry, [0])
        np.testing.assert_array_equal(result.negative_ancestry, [0, 0])

    def test_coplanar_triangle(self, worker):
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        plane_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        signs, dots = _cpu_classify(vertices, plane_point, plane_normal)
        assert all(s == 0 for s in signs)

        result = dispatch_clip_emit_reference(worker, vertices, faces, signs, dots)

        assert result.total_positive_faces == 1
        assert result.total_negative_faces == 1
        assert result.total_intersections == 0
        assert result.total_boundary_edges == 0
        np.testing.assert_array_equal(np.sort(result.positive_faces[0]), [0, 1, 2])
        np.testing.assert_array_equal(np.sort(result.negative_faces[0]), [0, 1, 2])

    def test_two_triangles_shared_edge(self, worker):
        # Two triangles sharing edge (v0, v1); v0 is above the plane and
        # v1 is below, so that shared edge is cut by both triangles.
        vertices = np.array(
            [
                [0.0, 1.0, 0.0],  # v0: above (shared cut-edge endpoint)
                [0.0, -1.0, 0.0],  # v1: below (shared cut-edge endpoint)
                [1.0, -1.0, 0.0],  # v2: below
                [-1.0, -1.0, 0.0],  # v3: below
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [0, 3, 1]], dtype=np.uint32)
        plane_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        signs, dots = _cpu_classify(vertices, plane_point, plane_normal)
        result = dispatch_clip_emit_reference(worker, vertices, faces, signs, dots)

        assert result.total_positive_faces == 2
        assert result.total_negative_faces == 4
        assert result.total_intersections == 4
        assert result.total_boundary_edges == 2

        # The shared edge (v0, v1) is cut identically by both triangles,
        # so its EdgeKey (0, 1) should appear exactly twice among the
        # intersection edges.
        edge_keys = [tuple(e) for e in result.intersection_edges]
        assert edge_keys.count((0, 1)) == 2

    def test_face_conservation(self, worker):
        # Tetrahedron: 4 faces.
        vertices = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0],
                [0.0, -1.0, -1.0],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 1],
                [1, 3, 2],
            ],
            dtype=np.uint32,
        )
        plane_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        signs, dots = _cpu_classify(vertices, plane_point, plane_normal)
        result = dispatch_clip_emit_reference(worker, vertices, faces, signs, dots)

        # Every source face must appear in exactly one of the ancestry
        # arrays for each output triangle it produced, and every face
        # contributes at least one output triangle to one side or the
        # other (nothing vanishes).
        all_ancestry = np.concatenate(
            [result.positive_ancestry, result.negative_ancestry]
        )
        seen_faces = set(all_ancestry.tolist())
        assert seen_faces <= set(range(len(faces)))
        for face_idx in range(len(faces)):
            s0, s1, s2 = signs[faces[face_idx]]
            has_pos = s0 > 0 or s1 > 0 or s2 > 0
            has_neg = s0 < 0 or s1 < 0 or s2 < 0
            if has_pos or has_neg:
                assert face_idx in seen_faces

    def test_sign_consistency(self, worker):
        vertices = np.array(
            [[0.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        plane_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v_count = len(vertices)

        signs, dots = _cpu_classify(vertices, plane_point, plane_normal)
        result = dispatch_clip_emit_reference(worker, vertices, faces, signs, dots)

        for tri in result.positive_faces:
            for vi in tri:
                if vi < v_count:
                    assert signs[vi] >= 0
        for tri in result.negative_faces:
            for vi in tri:
                if vi < v_count:
                    assert signs[vi] <= 0

    def test_empty_input(self, worker):
        vertices = np.zeros((0, 3), dtype=np.float32)
        faces = np.zeros((0, 3), dtype=np.uint32)
        signs = np.zeros(0, dtype=np.int32)
        dots = np.zeros(0, dtype=np.float32)

        result = dispatch_clip_emit_reference(worker, vertices, faces, signs, dots)

        assert result.total_positive_faces == 0
        assert result.total_negative_faces == 0
        assert result.total_intersections == 0
        assert result.total_boundary_edges == 0
        assert len(result.positive_faces) == 0
        assert len(result.negative_faces) == 0

    def test_winding_preservation(self, worker):
        # v0 above, v1/v2 below the y=0 plane; project onto XZ to check
        # winding via the cross product of edge vectors.
        vertices = np.array(
            [[0.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.5]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        plane_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        signs, dots = _cpu_classify(vertices, plane_point, plane_normal)
        result = dispatch_clip_emit_reference(worker, vertices, faces, signs, dots)

        v_count = len(vertices)

        def _position(vi: int) -> np.ndarray:
            if vi < v_count:
                return vertices[vi]
            return result.intersection_positions[vi - v_count]

        source_normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])

        for tri in np.concatenate([result.positive_faces, result.negative_faces]):
            p0, p1, p2 = (_position(int(vi)) for vi in tri)
            face_normal = np.cross(p1 - p0, p2 - p0)
            assert np.dot(face_normal, source_normal) > 0
