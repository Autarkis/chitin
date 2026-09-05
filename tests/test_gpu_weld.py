"""Intersection welding tests — EdgeKey dedup and index remap."""

import numpy as np
import pytest

from chitin.gpu.clip import ClipEmitResult
from chitin.gpu.weld import WeldResult, weld_intersections_reference


def _make_emit(
    pos_faces,
    neg_faces,
    ixn_positions,
    ixn_edges,
    boundary_edges,
    pos_ancestry,
    neg_ancestry,
):
    pf = (
        np.array(pos_faces, dtype=np.uint32).reshape(-1, 3)
        if pos_faces
        else np.zeros((0, 3), dtype=np.uint32)
    )
    nf = (
        np.array(neg_faces, dtype=np.uint32).reshape(-1, 3)
        if neg_faces
        else np.zeros((0, 3), dtype=np.uint32)
    )
    ip = (
        np.array(ixn_positions, dtype=np.float32).reshape(-1, 3)
        if ixn_positions
        else np.zeros((0, 3), dtype=np.float32)
    )
    ie = (
        np.array(ixn_edges, dtype=np.uint32).reshape(-1, 2)
        if ixn_edges
        else np.zeros((0, 2), dtype=np.uint32)
    )
    be = (
        np.array(boundary_edges, dtype=np.uint32).reshape(-1, 2)
        if boundary_edges
        else np.zeros((0, 2), dtype=np.uint32)
    )
    pa = (
        np.array(pos_ancestry, dtype=np.uint32)
        if pos_ancestry
        else np.array([], dtype=np.uint32)
    )
    na = (
        np.array(neg_ancestry, dtype=np.uint32)
        if neg_ancestry
        else np.array([], dtype=np.uint32)
    )
    return ClipEmitResult(
        positive_faces=pf,
        negative_faces=nf,
        intersection_positions=ip,
        intersection_edges=ie,
        boundary_edges=be,
        positive_ancestry=pa,
        negative_ancestry=na,
        total_positive_faces=len(pf),
        total_negative_faces=len(nf),
        total_intersections=len(ip),
        total_boundary_edges=len(be),
    )


@pytest.fixture(scope="module")
def worker():
    from chitin.gpu.worker import GPUWorker

    if not GPUWorker.available():
        pytest.skip("No GPU adapter available")
    with GPUWorker() as w:
        yield w


class TestWeldIntersectionsReference:
    def test_no_intersections(self):
        emit = _make_emit(
            pos_faces=[[0, 1, 2]],
            neg_faces=[[3, 4, 5]],
            ixn_positions=[],
            ixn_edges=[],
            boundary_edges=[],
            pos_ancestry=[0],
            neg_ancestry=[0],
        )
        result = weld_intersections_reference(emit, vertex_count=6)

        assert isinstance(result, WeldResult)
        assert result.intersection_count_raw == 0
        assert result.intersection_count_welded == 0
        np.testing.assert_array_equal(result.positive_faces, [[0, 1, 2]])
        np.testing.assert_array_equal(result.negative_faces, [[3, 4, 5]])
        assert len(result.welded_positions) == 0

    def test_unique_edges_no_welding(self):
        # Two intersections with different EdgeKeys — no welding occurs.
        emit = _make_emit(
            pos_faces=[[0, 4, 5]],  # ixn refs at V+0 and V+1
            neg_faces=[[1, 4, 5]],
            ixn_positions=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ixn_edges=[[0, 1], [2, 3]],  # different EdgeKeys
            boundary_edges=[[4, 5]],
            pos_ancestry=[0],
            neg_ancestry=[0],
        )
        result = weld_intersections_reference(emit, vertex_count=4)

        assert result.intersection_count_raw == 2
        assert result.intersection_count_welded == 2
        np.testing.assert_array_equal(result.positive_faces, [[0, 4, 5]])

    def test_duplicate_edges_welded(self):
        # Two triangles sharing cut edge (1, 2). Both produce an
        # intersection on that edge, which must weld to one vertex.
        #
        # EdgeKey (0, 1): raw 4 -> welded 0 -> global 4
        # EdgeKey (1, 2): raw 5 AND 6 -> welded 1 -> global 5
        # EdgeKey (1, 3): raw 7 -> welded 2 -> global 6
        emit = _make_emit(
            pos_faces=[[0, 4, 5], [3, 6, 7]],
            neg_faces=[[1, 5, 4], [2, 7, 6]],
            ixn_positions=[
                [0.5, 0.0, 0.0],  # raw 0: edge (0, 1)
                [0.0, 0.5, 0.0],  # raw 1: edge (1, 2)
                [0.0, 0.5, 0.0],  # raw 2: edge (1, 2) -- duplicate
                [0.0, 0.0, 0.5],  # raw 3: edge (1, 3)
            ],
            ixn_edges=[[0, 1], [1, 2], [1, 2], [1, 3]],
            boundary_edges=[[4, 5], [6, 7]],
            pos_ancestry=[0, 1],
            neg_ancestry=[0, 1],
        )
        result = weld_intersections_reference(emit, vertex_count=4)

        assert result.intersection_count_raw == 4
        assert result.intersection_count_welded == 3

        # Raw 4->V+0=4, raw 5->V+1=5, raw 6->V+1=5 (welded!), raw 7->V+2=6
        np.testing.assert_array_equal(result.positive_faces, [[0, 4, 5], [3, 5, 6]])
        np.testing.assert_array_equal(result.negative_faces, [[1, 5, 4], [2, 6, 5]])

        assert len(result.welded_positions) == 3
        np.testing.assert_allclose(result.welded_positions[0], [0.5, 0.0, 0.0])
        np.testing.assert_allclose(result.welded_positions[1], [0.0, 0.5, 0.0])
        np.testing.assert_allclose(result.welded_positions[2], [0.0, 0.0, 0.5])

        # Boundary edges remapped: (4,5)->(4,5), (6,7)->(5,6)
        np.testing.assert_array_equal(result.boundary_edges, [[4, 5], [5, 6]])

    def test_source_indices_unchanged(self):
        emit = _make_emit(
            pos_faces=[[0, 4, 5], [3, 6, 7]],
            neg_faces=[[1, 5, 4], [2, 7, 6]],
            ixn_positions=[
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
            ],
            ixn_edges=[[0, 1], [1, 2], [1, 2], [1, 3]],
            boundary_edges=[[4, 5], [6, 7]],
            pos_ancestry=[0, 1],
            neg_ancestry=[0, 1],
        )
        result = weld_intersections_reference(emit, vertex_count=4)

        # Source vertex indices (< V) are never modified by welding.
        src_mask = result.positive_faces < 4
        np.testing.assert_array_equal(
            result.positive_faces[src_mask],
            emit.positive_faces[src_mask],
        )
        src_mask_neg = result.negative_faces < 4
        np.testing.assert_array_equal(
            result.negative_faces[src_mask_neg],
            emit.negative_faces[src_mask_neg],
        )

    def test_ancestry_preserved(self):
        emit = _make_emit(
            pos_faces=[[0, 4, 5], [3, 6, 7]],
            neg_faces=[[1, 5, 4], [2, 7, 6]],
            ixn_positions=[
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
            ],
            ixn_edges=[[0, 1], [1, 2], [1, 2], [1, 3]],
            boundary_edges=[[4, 5], [6, 7]],
            pos_ancestry=[0, 1],
            neg_ancestry=[0, 1],
        )
        result = weld_intersections_reference(emit, vertex_count=4)

        np.testing.assert_array_equal(result.positive_ancestry, [0, 1])
        np.testing.assert_array_equal(result.negative_ancestry, [0, 1])

    def test_welded_leq_raw(self):
        # Invariant: welded count never exceeds raw count.
        emit = _make_emit(
            pos_faces=[[0, 4, 5], [3, 6, 7]],
            neg_faces=[[1, 5, 4], [2, 7, 6]],
            ixn_positions=[
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
            ],
            ixn_edges=[[0, 1], [1, 2], [1, 2], [1, 3]],
            boundary_edges=[[4, 5], [6, 7]],
            pos_ancestry=[0, 1],
            neg_ancestry=[0, 1],
        )
        result = weld_intersections_reference(emit, vertex_count=4)

        assert result.intersection_count_welded <= result.intersection_count_raw

    def test_empty_faces(self):
        # Empty faces but some intersections -- shouldn't happen in
        # practice, but the function should handle it gracefully.
        emit = _make_emit(
            pos_faces=[],
            neg_faces=[],
            ixn_positions=[[1.0, 0.0, 0.0]],
            ixn_edges=[[0, 1]],
            boundary_edges=[],
            pos_ancestry=[],
            neg_ancestry=[],
        )
        result = weld_intersections_reference(emit, vertex_count=4)

        assert result.intersection_count_welded == 1
        assert len(result.positive_faces) == 0
        assert len(result.negative_faces) == 0

    def test_gpu_end_to_end(self, worker):
        """Classify + clip + weld on a real mesh."""
        from chitin.gpu.classify import dispatch_classify
        from chitin.gpu.clip import dispatch_clip_emit_reference

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

        signs, dots = dispatch_classify(worker, vertices, plane_point, plane_normal)
        emit = dispatch_clip_emit_reference(worker, vertices, faces, signs, dots)
        result = weld_intersections_reference(emit, vertex_count=len(vertices))

        # Both triangles cut the shared edge (v0, v1) -> EdgeKey (0, 1)
        # appears twice among the raw intersections -> welding should
        # reduce the intersection count.
        assert result.intersection_count_welded < result.intersection_count_raw
        assert result.intersection_count_welded > 0

        # All face indices should be valid.
        max_idx = len(vertices) + result.intersection_count_welded - 1
        assert result.positive_faces.max() <= max_idx
        assert result.negative_faces.max() <= max_idx
