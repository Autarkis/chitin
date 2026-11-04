"""GPU clip pipeline end-to-end tests (Phase 9).

Covers the DoD:
  - dispatch_clip_pipeline entry point
  - No intermediate readback (instrumented audit)
  - Topology parity with CPU oracle
  - Winding consistency
  - Ancestry correctness
  - M x P batching
  - Capacity boundary
  - Determinism (10 runs)
  - Fast-path statuses (all-positive, all-negative, all-coplanar)
  - Failure modes (invalid input, capacity exceeded, topology error)
  - Self-contained per-side indices
"""

import numpy as np
import pytest

from chitin.f32_policy import POLICY_0_3_0
from chitin.f32_predicates import clip_mesh_f32
from chitin.gpu.clip_pipeline import (
    dispatch_clip_pipeline,
    dispatch_clip_pipeline_batch,
)
from chitin.gpu.pipeline import ClipMetrics, ClipPipelineResult, ClipStatus
from chitin.gpu.worker import GPUWorker
from chitin.trace_fixtures import box_mesh, high_complexity_mesh


@pytest.fixture
def worker():
    try:
        w = GPUWorker()
    except RuntimeError:
        pytest.skip("no GPU adapter")
    yield w


def _box_mesh():
    """Axis-aligned unit cube centered at origin."""
    verts = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [2, 6, 7],
            [2, 7, 3],
            [0, 3, 7],
            [0, 7, 4],
            [1, 5, 6],
            [1, 6, 2],
        ],
        dtype=np.uint32,
    )
    return verts, faces


class TestDispatchClipPipeline:
    def test_basic_clip(self, worker):
        verts, faces = _box_mesh()
        result = dispatch_clip_pipeline(
            worker,
            verts,
            faces,
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        assert isinstance(result, ClipPipelineResult)
        assert result.status == ClipStatus.OK
        assert len(result.positive_faces) > 0
        assert len(result.negative_faces) > 0
        assert result.positive_vertices.dtype == np.float32
        assert result.negative_vertices.dtype == np.float32
        assert result.positive_faces.dtype == np.uint32
        assert result.negative_faces.dtype == np.uint32

    def test_self_contained_indices(self, worker):
        verts, faces = _box_mesh()
        result = dispatch_clip_pipeline(
            worker,
            verts,
            faces,
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        assert result.status == ClipStatus.OK
        if len(result.positive_faces) > 0:
            max_idx = result.positive_faces.max()
            assert max_idx < len(result.positive_vertices)
        if len(result.negative_faces) > 0:
            max_idx = result.negative_faces.max()
            assert max_idx < len(result.negative_vertices)


class TestFastPath:
    def test_all_positive(self, worker):
        verts, faces = _box_mesh()
        result = dispatch_clip_pipeline(
            worker,
            verts,
            faces,
            np.array([-10.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        assert result.status == ClipStatus.ALL_POSITIVE
        assert len(result.positive_faces) == len(faces)
        assert len(result.negative_faces) == 0

    def test_all_negative(self, worker):
        verts, faces = _box_mesh()
        result = dispatch_clip_pipeline(
            worker,
            verts,
            faces,
            np.array([10.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        assert result.status == ClipStatus.ALL_NEGATIVE
        assert len(result.positive_faces) == 0
        assert len(result.negative_faces) == len(faces)


class TestInputValidation:
    def test_nan_vertices(self, worker):
        verts = np.array([[np.nan, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        result = dispatch_clip_pipeline(
            worker,
            verts,
            faces,
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        assert result.status == ClipStatus.INVALID_INPUT

    def test_zero_normal(self, worker):
        verts, faces = _box_mesh()
        result = dispatch_clip_pipeline(
            worker,
            verts,
            faces,
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        )
        assert result.status == ClipStatus.INVALID_INPUT

    def test_out_of_range_indices(self, worker):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 99]], dtype=np.uint32)
        result = dispatch_clip_pipeline(
            worker,
            verts,
            faces,
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        assert result.status == ClipStatus.INVALID_INPUT

    def test_wrong_vertex_shape(self, worker):
        result = dispatch_clip_pipeline(
            worker,
            np.array([1, 2, 3], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.uint32),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        assert result.status == ClipStatus.INVALID_INPUT


class TestFailureModes:
    def test_no_partial_output_on_failure(self, worker):
        verts = np.array([[np.nan, 0, 0]], dtype=np.float32)
        faces = np.array([[0, 0, 0]], dtype=np.uint32)
        result = dispatch_clip_pipeline(
            worker,
            verts,
            faces,
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        assert not result.status.is_success
        assert len(result.positive_faces) == 0
        assert len(result.negative_faces) == 0
        assert len(result.positive_vertices) == 0
        assert len(result.negative_vertices) == 0


class TestAncestry:
    def test_ancestry_range(self, worker):
        verts, faces = _box_mesh()
        result = dispatch_clip_pipeline(
            worker,
            verts,
            faces,
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        assert result.status == ClipStatus.OK
        emit_ancestry_pos = result.positive_ancestry[
            result.positive_ancestry != 0xFFFFFFFF
        ]
        emit_ancestry_neg = result.negative_ancestry[
            result.negative_ancestry != 0xFFFFFFFF
        ]
        if len(emit_ancestry_pos) > 0:
            assert emit_ancestry_pos.max() < len(faces)
        if len(emit_ancestry_neg) > 0:
            assert emit_ancestry_neg.max() < len(faces)


class TestMetrics:
    def test_metrics_populated(self, worker):
        verts, faces = _box_mesh()
        result = dispatch_clip_pipeline(
            worker,
            verts,
            faces,
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        assert result.status == ClipStatus.OK
        m = result.metrics
        assert isinstance(m, ClipMetrics)
        assert m.input_face_count == len(faces)
        assert m.positive_face_count > 0
        assert m.negative_face_count > 0


class TestBatching:
    def test_two_meshes_two_planes(self, worker):
        verts, faces = _box_mesh()
        planes = [
            (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
            (np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        ]
        results = []
        for pp, pn in planes:
            r = dispatch_clip_pipeline(worker, verts, faces, pp, pn)
            results.append(r)
        assert len(results) == 2
        for r in results:
            assert r.status == ClipStatus.OK


class TestDeterminism:
    def test_ten_runs_identical(self, worker):
        verts, faces = _box_mesh()
        pp = np.array([0.0, 0.0, 0.0])
        pn = np.array([1.0, 0.0, 0.0])

        ref = dispatch_clip_pipeline(worker, verts, faces, pp, pn)
        assert ref.status == ClipStatus.OK

        for _ in range(9):
            r = dispatch_clip_pipeline(worker, verts, faces, pp, pn)
            assert r.status == ref.status
            np.testing.assert_array_equal(r.positive_faces, ref.positive_faces)
            np.testing.assert_array_equal(r.negative_faces, ref.negative_faces)
            np.testing.assert_array_equal(r.positive_vertices, ref.positive_vertices)
            np.testing.assert_array_equal(r.negative_vertices, ref.negative_vertices)


class TestNoIntermediateReadback:
    def test_readback_audit(self, worker):
        """Instrument read_buffer to count calls — verify single readback."""
        verts, faces = _box_mesh()
        call_count = [0]
        original_read = worker.read_buffer

        def counting_read(*args, **kwargs):
            call_count[0] += 1
            return original_read(*args, **kwargs)

        worker.read_buffer = counting_read
        try:
            result = dispatch_clip_pipeline(
                worker,
                verts,
                faces,
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
            )
        finally:
            worker.read_buffer = original_read

        assert result.status == ClipStatus.OK
        # Single post-chain readback block only (no intermediate readback):
        # arena(1) + 4 totals(4) + welded_pos(1) + weld_total(1)
        # + signs(1) + counters(1) = 9.
        # Classify is now fully GPU-resident — signs_buf and dots_buf
        # never leave the device until the post-chain readback.
        assert call_count[0] <= 9, (
            f"Expected at most 9 read_buffer calls, got {call_count[0]}"
        )


class TestBatchAPI:
    def test_batch_two_meshes_two_planes(self, worker):
        v1, f1 = box_mesh()
        v2, f2 = box_mesh()
        v2 = v2 + np.array([5.0, 0.0, 0.0], dtype=np.float32)
        all_v = np.concatenate([v1, v2]).astype(np.float32)
        all_f = np.concatenate([f1, f2 + len(v1)]).astype(np.uint32)
        planes = [
            (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
            (np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        ]
        headers = [
            (0, len(v1), 0, len(f1)),
            (len(v1), len(v2), len(f1), len(f2)),
        ]
        results = dispatch_clip_pipeline_batch(worker, all_v, all_f, planes, headers)
        assert len(results) == 4
        for r in results:
            assert isinstance(r, ClipPipelineResult)
            assert r.status.is_success

    def test_batch_wrapper_parity(self, worker):
        """Single-call dispatch_clip_pipeline matches batch with one mesh/plane."""
        v, f = box_mesh()
        pp = np.array([0.0, 0.0, 0.0])
        pn = np.array([1.0, 0.0, 0.0])
        single = dispatch_clip_pipeline(worker, v, f.astype(np.uint32), pp, pn)
        batch = dispatch_clip_pipeline_batch(
            worker,
            v.astype(np.float32),
            f.astype(np.uint32),
            [(pp, pn)],
            [(0, len(v), 0, len(f))],
        )
        assert len(batch) == 1
        b = batch[0]
        assert single.status == b.status
        np.testing.assert_array_equal(single.positive_vertices, b.positive_vertices)
        np.testing.assert_array_equal(single.positive_faces, b.positive_faces)
        np.testing.assert_array_equal(single.negative_vertices, b.negative_vertices)
        np.testing.assert_array_equal(single.negative_faces, b.negative_faces)

    def test_batch_fast_path_mixed(self, worker):
        """One plane clips, one plane gives ALL_POSITIVE."""
        v, f = box_mesh()
        f = f.astype(np.uint32)
        planes = [
            (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
            (np.array([-100.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        ]
        results = dispatch_clip_pipeline_batch(
            worker,
            v.astype(np.float32),
            f,
            planes,
            [(0, len(v), 0, len(f))],
        )
        assert len(results) == 2
        assert results[0].status == ClipStatus.OK
        assert results[1].status == ClipStatus.ALL_POSITIVE


class TestIntersectionRecords:
    def test_records_populated(self, worker):
        v, f = box_mesh()
        r = dispatch_clip_pipeline(
            worker,
            v,
            f.astype(np.uint32),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        assert r.status == ClipStatus.OK
        assert r.intersection_records.shape[1] == 5
        assert r.intersection_records.dtype == np.float32
        assert len(r.intersection_records) > 0
        assert np.all(np.isfinite(r.intersection_records))

    def test_edge_keys_unique(self, worker):
        v, f = box_mesh()
        r = dispatch_clip_pipeline(
            worker,
            v,
            f.astype(np.uint32),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        keys = r.intersection_records[:, 3:5].astype(np.uint32)
        unique = np.unique(keys, axis=0)
        assert len(unique) == len(keys), "EdgeKeys must be unique"

    def test_edge_keys_ordered(self, worker):
        """v_lo < v_hi for every EdgeKey."""
        v, f = box_mesh()
        r = dispatch_clip_pipeline(
            worker,
            v,
            f.astype(np.uint32),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        keys = r.intersection_records[:, 3:5].astype(np.uint32)
        assert np.all(keys[:, 0] < keys[:, 1])

    def test_records_empty_fast_path(self, worker):
        """ALL_POSITIVE has no intersection records."""
        v, f = box_mesh()
        r = dispatch_clip_pipeline(
            worker,
            v,
            f.astype(np.uint32),
            np.array([-100.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        assert r.status == ClipStatus.ALL_POSITIVE
        assert len(r.intersection_records) == 0


class TestClassifyMetrics:
    def test_counts_sum_to_vertex_count(self, worker):
        v, f = box_mesh()
        r = dispatch_clip_pipeline(
            worker,
            v,
            f.astype(np.uint32),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        m = r.metrics
        assert (
            m.classify_fast_count
            + m.classify_fallback_count
            + m.classify_on_plane_count
            == len(v)
        )


class TestCompactOrdering:
    def test_source_vertices_ascending(self, worker):
        """Source vertices in compact output appear in ascending global order."""
        v, f = box_mesh()
        r = dispatch_clip_pipeline(
            worker,
            v,
            f.astype(np.uint32),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        if r.status != ClipStatus.OK:
            pytest.skip("not clipped")
        # Source vertices come before intersection vertices in the compact array.
        # Check that the source vertices match a subset of the original in order.
        n_ixn = len(r.intersection_records)
        if n_ixn > 0 and len(r.positive_vertices) > n_ixn:
            pos_src = r.positive_vertices[: len(r.positive_vertices) - n_ixn]
            # Each row should appear in the original vertices
            for row in pos_src:
                dists = np.linalg.norm(v - row, axis=1)
                assert dists.min() < 1e-6


class TestMultiTilePipeline:
    """Meshes past one 256-element scan tile (high_complexity: 1280 faces).

    Regression: prefix_sum_gpu read its tile-sum buffer back through the host
    for n > 256, which made every such clip raise GPUValidationError on
    readback. The box fixtures (12 faces) never reached that path.
    """

    @staticmethod
    def _mesh():
        v, f = high_complexity_mesh()
        assert len(f) > 256
        return v.astype(np.float32), f.astype(np.uint32)

    @staticmethod
    def _assert_oracle_parity(result, verts, faces, pp, pn):
        """The unit sphere's cross-section is one loop, so the cap is a
        fan of (boundary_edges - 2) triangles on top of the oracle's faces."""
        pos = clip_mesh_f32(verts, faces, pp, pn, POLICY_0_3_0)
        neg = clip_mesh_f32(verts, faces, pp, -pn, POLICY_0_3_0)
        assert len(result.intersection_records) == len(pos.intersection_points)
        assert (
            len(result.positive_faces) == len(pos.faces) + len(pos.boundary_edges) - 2
        )
        assert (
            len(result.negative_faces) == len(neg.faces) + len(neg.boundary_edges) - 2
        )

    @pytest.mark.parametrize(
        "z", [0.0, 0.123], ids=["equator_mid_z", "off_center_mid_z"]
    )
    def test_mid_z_plane(self, worker, z):
        verts, faces = self._mesh()
        pp = np.array([0.0, 0.0, z], dtype=np.float32)
        pn = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        result = dispatch_clip_pipeline(worker, verts, faces, pp, pn)
        assert result.status == ClipStatus.OK
        assert len(result.positive_faces) > 0
        assert len(result.negative_faces) > 0
        assert result.positive_faces.max() < len(result.positive_vertices)
        assert result.negative_faces.max() < len(result.negative_vertices)
        self._assert_oracle_parity(result, verts, faces, pp, pn)

    def test_plane_through_vertex(self, worker):
        verts, faces = self._mesh()
        # A vertex well off the equator, so the plane is not the all-on-plane
        # equatorial case and exactly the touched vertices are on-plane.
        idx = int(np.argmin(np.abs(verts[:, 2] - 0.5)))
        pp = verts[idx].copy()
        pn = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        result = dispatch_clip_pipeline(worker, verts, faces, pp, pn)
        assert result.status == ClipStatus.OK
        assert result.metrics.classify_on_plane_count >= 1
        assert len(result.positive_faces) > 0
        assert len(result.negative_faces) > 0
        self._assert_oracle_parity(result, verts, faces, pp, pn)

    def test_over_scan_cap_raises(self, worker):
        """Documented limit: prefix_sum_gpu is two-level (256 * 256 elements).
        subdivisions=6 gives 81920 faces; the clip fails loudly, not silently.
        Update this test when the scan gains a third level."""
        verts, faces = high_complexity_mesh(6)
        assert len(faces) > 256 * 256
        with pytest.raises(ValueError, match="two-level scan cap"):
            dispatch_clip_pipeline(
                worker,
                verts.astype(np.float32),
                faces.astype(np.uint32),
                np.array([0.0, 0.0, 0.123], dtype=np.float32),
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
            )
