"""Boundary loop extraction tests."""

import numpy as np
import pytest

from chitin.gpu.loops import extract_boundary_loops
from chitin.gpu.weld import WeldResult


def _make_weld_result(boundary_edges) -> WeldResult:
    """Minimal WeldResult with just boundary edges populated."""
    be = (
        np.array(boundary_edges, dtype=np.uint32).reshape(-1, 2)
        if boundary_edges
        else np.zeros((0, 2), dtype=np.uint32)
    )
    return WeldResult(
        positive_faces=np.zeros((0, 3), dtype=np.uint32),
        negative_faces=np.zeros((0, 3), dtype=np.uint32),
        welded_positions=np.zeros((0, 3), dtype=np.float32),
        welded_edge_keys=np.zeros((0, 2), dtype=np.uint32),
        boundary_edges=be,
        positive_ancestry=np.array([], dtype=np.uint32),
        negative_ancestry=np.array([], dtype=np.uint32),
        intersection_count_raw=0,
        intersection_count_welded=0,
    )


class TestExtractBoundaryLoops:
    def test_empty_boundary(self):
        """No boundary edges produces no loops, valid=True."""
        result = extract_boundary_loops(_make_weld_result([]))
        assert result.valid is True
        assert result.loops == []

    def test_single_triangle_loop(self):
        """Three edges forming a triangle produce one 3-vertex loop."""
        weld = _make_weld_result([[0, 1], [1, 2], [2, 0]])
        result = extract_boundary_loops(weld)
        assert result.valid is True
        assert len(result.loops) == 1
        assert len(result.loops[0]) == 3
        assert set(result.loops[0].tolist()) == {0, 1, 2}

    def test_quad_loop(self):
        """Four edges forming a quad produce one 4-vertex loop."""
        weld = _make_weld_result([[0, 1], [1, 2], [2, 3], [3, 0]])
        result = extract_boundary_loops(weld)
        assert result.valid is True
        assert len(result.loops) == 1
        assert len(result.loops[0]) == 4
        assert set(result.loops[0].tolist()) == {0, 1, 2, 3}

    def test_two_disconnected_loops(self):
        """Two separate triangles produce two loops."""
        weld = _make_weld_result(
            [
                [0, 1],
                [1, 2],
                [2, 0],
                [10, 11],
                [11, 12],
                [12, 10],
            ]
        )
        result = extract_boundary_loops(weld)
        assert result.valid is True
        assert len(result.loops) == 2
        sets = [set(loop.tolist()) for loop in result.loops]
        assert {0, 1, 2} in sets
        assert {10, 11, 12} in sets

    def test_valence_violation_branching(self):
        """Vertex with 3+ neighbors (branching) produces valid=False."""
        weld = _make_weld_result([[0, 1], [1, 2], [1, 3]])
        result = extract_boundary_loops(weld)
        assert result.valid is False
        assert result.loops == []

    def test_valence_violation_dangling(self):
        """Vertex with valence 1 (dangling edge) produces valid=False."""
        weld = _make_weld_result([[0, 1]])
        result = extract_boundary_loops(weld)
        assert result.valid is False
        assert result.loops == []

    def test_deterministic_output_order(self):
        """Same edges in different input orders produce identical loops."""
        edges_a = [[0, 1], [1, 2], [2, 3], [3, 0]]
        edges_b = [[3, 0], [1, 2], [0, 1], [2, 3]]
        edges_c = [[2, 1], [0, 3], [3, 2], [1, 0]]

        results = [
            extract_boundary_loops(_make_weld_result(e))
            for e in [edges_a, edges_b, edges_c]
        ]

        for r in results:
            assert r.valid is True
            assert len(r.loops) == 1

        ref = results[0].loops[0].tolist()
        for r in results[1:]:
            assert r.loops[0].tolist() == ref

    def test_deterministic_ten_runs(self):
        """Ten consecutive runs on same input produce identical output."""
        edges = [[5, 10], [10, 15], [15, 20], [20, 25], [25, 5]]
        weld = _make_weld_result(edges)

        ref = extract_boundary_loops(weld)
        for _ in range(9):
            result = extract_boundary_loops(weld)
            assert len(result.loops) == len(ref.loops)
            for a, b in zip(ref.loops, result.loops):
                np.testing.assert_array_equal(a, b)

    def test_large_loop_above_256_edges(self):
        """Loop with >256 boundary edges extracts correctly."""
        n = 300
        edges = [[i, (i + 1) % n] for i in range(n)]
        weld = _make_weld_result(edges)

        result = extract_boundary_loops(weld)
        assert result.valid is True
        assert len(result.loops) == 1
        assert len(result.loops[0]) == n

    def test_intersection_vertex_indices(self):
        """Loops correctly include intersection vertices (indices >= vertex_count)."""
        weld = _make_weld_result(
            [
                [0, 100],
                [100, 1],
                [1, 101],
                [101, 2],
                [2, 102],
                [102, 0],
            ]
        )
        result = extract_boundary_loops(weld)
        assert result.valid is True
        assert len(result.loops) == 1
        assert len(result.loops[0]) == 6
        verts = set(result.loops[0].tolist())
        assert verts == {0, 1, 2, 100, 101, 102}

    def test_loop_is_closed(self):
        """Each extracted loop forms a closed ring (last connects to first)."""
        weld = _make_weld_result([[0, 1], [1, 2], [2, 3], [3, 0]])
        result = extract_boundary_loops(weld)
        assert result.valid is True
        loop = result.loops[0]
        loop_edges = set()
        for i in range(len(loop)):
            a, b = int(loop[i]), int(loop[(i + 1) % len(loop)])
            loop_edges.add(frozenset((a, b)))
        for edge in [[0, 1], [1, 2], [2, 3], [3, 0]]:
            assert frozenset(edge) in loop_edges

    def test_result_dtype(self):
        """Loop arrays have uint32 dtype."""
        weld = _make_weld_result([[0, 1], [1, 2], [2, 0]])
        result = extract_boundary_loops(weld)
        assert result.valid is True
        for loop in result.loops:
            assert loop.dtype == np.uint32

    def test_cpu_oracle_parity_single_split(self):
        """A lone (non-watertight) triangle's cut chord is a dangling edge.

        clip_emit produces one boundary edge between the two cut points
        (boundary_edges == [[3, 4]]); each endpoint has valence 1 since
        there is no sibling triangle to close the ring. This matches
        test_valence_violation_dangling's semantics exactly -- capping
        a genuinely closed loop requires a watertight input, which a
        single triangle is not (see test_cpu_oracle_parity_tetrahedron).
        """
        from chitin.gpu.worker import GPUWorker

        if not GPUWorker.available():
            pytest.skip("No GPU adapter available")

        vertices = np.array(
            [[0.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        plane_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        from chitin.gpu.classify import dispatch_classify
        from chitin.gpu.clip import dispatch_clip_emit
        from chitin.gpu.weld import weld_intersections

        with GPUWorker() as worker:
            signs, dots = dispatch_classify(worker, vertices, plane_point, plane_normal)
            emit = dispatch_clip_emit(worker, vertices, faces, signs, dots)
            weld = weld_intersections(emit, vertex_count=len(vertices))
            gpu_result = extract_boundary_loops(weld)

        assert weld.boundary_edges.shape[0] == 1
        assert gpu_result.valid is False
        assert gpu_result.loops == []

    def test_cpu_oracle_parity_tetrahedron(self):
        """Tetrahedron slice produces valid loops matching topology expectations."""
        from chitin.gpu.worker import GPUWorker

        if not GPUWorker.available():
            pytest.skip("No GPU adapter available")

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

        from chitin.gpu.classify import dispatch_classify
        from chitin.gpu.clip import dispatch_clip_emit
        from chitin.gpu.weld import weld_intersections

        with GPUWorker() as worker:
            signs, dots = dispatch_classify(worker, vertices, plane_point, plane_normal)
            emit = dispatch_clip_emit(worker, vertices, faces, signs, dots)
            weld = weld_intersections(emit, vertex_count=len(vertices))
            result = extract_boundary_loops(weld)

        assert result.valid is True
        assert len(result.loops) >= 1
        total_verts = sum(len(loop) for loop in result.loops)
        assert total_verts == weld.boundary_edges.shape[0]
