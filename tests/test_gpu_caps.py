"""Cap triangulation tests."""

import numpy as np
import pytest

from chitin.gpu.caps import triangulate_caps
from chitin.gpu.weld import WeldResult


def _make_weld_result(welded_positions=None) -> WeldResult:
    """Minimal WeldResult with just welded_positions populated."""
    wp = (
        np.array(welded_positions, dtype=np.float32).reshape(-1, 3)
        if welded_positions
        else np.zeros((0, 3), dtype=np.float32)
    )
    return WeldResult(
        positive_faces=np.zeros((0, 3), dtype=np.uint32),
        negative_faces=np.zeros((0, 3), dtype=np.uint32),
        welded_positions=wp,
        welded_edge_keys=np.zeros((0, 2), dtype=np.uint32),
        boundary_edges=np.zeros((0, 2), dtype=np.uint32),
        positive_ancestry=np.array([], dtype=np.uint32),
        negative_ancestry=np.array([], dtype=np.uint32),
        intersection_count_raw=0,
        intersection_count_welded=0,
    )


def _face_normal(verts: np.ndarray, face) -> np.ndarray:
    a, b, c = verts[face[0]], verts[face[1]], verts[face[2]]
    return np.cross(b - a, c - a)


class TestTriangulateCaps:
    def test_empty_loops(self):
        """No loops produces an invalid, empty result."""
        source = np.zeros((0, 3), dtype=np.float32)
        result = triangulate_caps([], _make_weld_result(), source, np.array([0, 1, 0]))
        assert result.valid is False
        assert result.loop_count == 0
        assert result.total_cap_faces == 0
        assert result.positive_cap_faces.shape == (0, 3)
        assert result.negative_cap_faces.shape == (0, 3)

    def test_invalid_loop_too_short(self):
        """A loop with fewer than 3 vertices makes the whole result invalid."""
        source = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        loops = [np.array([0, 1], dtype=np.uint32)]
        result = triangulate_caps(
            loops, _make_weld_result(), source, np.array([0, 1, 0])
        )
        assert result.valid is False
        assert result.total_cap_faces == 0

    def test_triangle_loop(self):
        """A single triangle loop produces exactly one face per side."""
        source = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        loops = [np.array([0, 1, 2], dtype=np.uint32)]
        result = triangulate_caps(
            loops, _make_weld_result(), source, np.array([0.0, 1.0, 0.0])
        )
        assert result.valid is True
        assert result.loop_count == 1
        assert result.total_cap_faces == 1
        assert result.positive_cap_faces.shape == (1, 3)
        assert result.negative_cap_faces.shape == (1, 3)
        assert set(result.positive_cap_faces[0].tolist()) == {0, 1, 2}

    def test_quad_loop(self):
        """A single quad loop produces exactly two faces per side."""
        source = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        loops = [np.array([0, 1, 2, 3], dtype=np.uint32)]
        result = triangulate_caps(
            loops, _make_weld_result(), source, np.array([0.0, 1.0, 0.0])
        )
        assert result.valid is True
        assert result.total_cap_faces == 2
        assert result.positive_cap_faces.shape == (2, 3)
        assert result.negative_cap_faces.shape == (2, 3)
        covered = set()
        for face in result.positive_cap_faces:
            covered.update(face.tolist())
        assert covered == {0, 1, 2, 3}

    def test_opposite_winding(self):
        """Negative cap faces are the positive faces with columns 1,2 swapped."""
        source = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        loops = [np.array([0, 1, 2], dtype=np.uint32)]
        result = triangulate_caps(
            loops, _make_weld_result(), source, np.array([0.0, 1.0, 0.0])
        )
        expected_negative = result.positive_cap_faces[:, [0, 2, 1]]
        np.testing.assert_array_equal(result.negative_cap_faces, expected_negative)

    def test_cap_normal_opposition(self):
        """Positive and negative cap faces have oppositely-directed normals."""
        source = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        loops = [np.array([0, 1, 2, 3], dtype=np.uint32)]
        result = triangulate_caps(
            loops, _make_weld_result(), source, np.array([0.0, 1.0, 0.0])
        )
        for pos_face, neg_face in zip(
            result.positive_cap_faces, result.negative_cap_faces
        ):
            pos_normal = _face_normal(source, pos_face)
            neg_normal = _face_normal(source, neg_face)
            assert np.dot(pos_normal, neg_normal) < 0

    def test_multiple_loops(self):
        """Two separate triangle loops each contribute one face."""
        source = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [10.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        loops = [
            np.array([0, 1, 2], dtype=np.uint32),
            np.array([3, 4, 5], dtype=np.uint32),
        ]
        result = triangulate_caps(
            loops, _make_weld_result(), source, np.array([0.0, 1.0, 0.0])
        )
        assert result.valid is True
        assert result.loop_count == 2
        assert result.total_cap_faces == 2

    def test_intersection_vertex_references(self):
        """Loop indices >= len(source_vertices) resolve into welded_positions."""
        source = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        # index 2 is the first welded/intersection vertex (offset by
        # len(source_vertices), matching chitin.gpu.weld._remap_ixn).
        weld = _make_weld_result([[0.0, 0.0, 1.0]])
        loops = [np.array([0, 1, 2], dtype=np.uint32)]
        result = triangulate_caps(loops, weld, source, np.array([0.0, 1.0, 0.0]))
        assert result.valid is True
        assert result.total_cap_faces == 1
        assert set(result.positive_cap_faces[0].tolist()) == {0, 1, 2}

    def test_result_dtype(self):
        """Cap face arrays have uint32 dtype."""
        source = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        loops = [np.array([0, 1, 2], dtype=np.uint32)]
        result = triangulate_caps(
            loops, _make_weld_result(), source, np.array([0.0, 1.0, 0.0])
        )
        assert result.positive_cap_faces.dtype == np.uint32
        assert result.negative_cap_faces.dtype == np.uint32

    def test_deterministic_ten_runs(self):
        """Ten consecutive runs on the same input produce identical faces."""
        source = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.2, 0.0, 0.8],
                [0.9, 0.0, 1.3],
                [0.1, 0.0, 1.1],
            ],
            dtype=np.float32,
        )
        loops = [np.array([0, 1, 2, 3, 4], dtype=np.uint32)]
        plane_normal = np.array([0.0, 1.0, 0.0])

        ref = triangulate_caps(loops, _make_weld_result(), source, plane_normal)
        for _ in range(9):
            result = triangulate_caps(loops, _make_weld_result(), source, plane_normal)
            np.testing.assert_array_equal(
                result.positive_cap_faces, ref.positive_cap_faces
            )
            np.testing.assert_array_equal(
                result.negative_cap_faces, ref.negative_cap_faces
            )

    def test_gpu_end_to_end_tetrahedron(self):
        """Full pipeline: classify -> clip -> weld -> loops -> triangulate_caps."""
        from chitin.gpu.worker import GPUWorker

        if not GPUWorker.available():
            pytest.skip("No GPU adapter available")

        from chitin.gpu.classify import dispatch_classify
        from chitin.gpu.clip import dispatch_clip_emit
        from chitin.gpu.loops import extract_boundary_loops
        from chitin.gpu.weld import weld_intersections

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
            [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
            dtype=np.uint32,
        )
        plane_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        with GPUWorker() as worker:
            signs, dots = dispatch_classify(worker, vertices, plane_point, plane_normal)
            emit = dispatch_clip_emit(worker, vertices, faces, signs, dots)
            weld = weld_intersections(emit, vertex_count=len(vertices))
            loop_result = extract_boundary_loops(weld)

        assert loop_result.valid is True

        cap = triangulate_caps(loop_result.loops, weld, vertices, plane_normal)
        assert cap.valid is True
        assert cap.loop_count == len(loop_result.loops)
        assert cap.total_cap_faces > 0
        assert cap.positive_cap_faces.dtype == np.uint32
        assert cap.negative_cap_faces.dtype == np.uint32
