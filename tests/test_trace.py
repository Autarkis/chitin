"""Trace recorder and oracle replay tests."""

from __future__ import annotations

import json

import numpy as np

from chitin.trace import TraceRecorder, _geometry_digest, _hull_digest


def _box_arrays():
    vertices = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 5, 7],
            [4, 7, 6],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 3, 7],
            [1, 7, 5],
        ],
        dtype=np.int32,
    )
    return vertices, faces


class TestGeometryDigest:
    def test_deterministic(self):
        v, f = _box_arrays()
        assert _geometry_digest(v, f) == _geometry_digest(v, f)

    def test_different_geometry_different_digest(self):
        v, f = _box_arrays()
        v2 = v * 2.0
        assert _geometry_digest(v, f) != _geometry_digest(v2, f)

    def test_format_prefix(self):
        v, f = _box_arrays()
        assert _geometry_digest(v, f).startswith("sha256:")

    def test_vertices_only(self):
        v, _ = _box_arrays()
        d = _geometry_digest(v)
        assert d.startswith("sha256:")

    def test_faces_affect_digest(self):
        v, f = _box_arrays()
        f2 = f.copy()
        f2[0, 0] = 7
        assert _geometry_digest(v, f) != _geometry_digest(v, f2)


class TestHullDigest:
    def test_deterministic(self):
        v, f = _box_arrays()
        hulls = [(v, f[:, :3].ravel().astype(np.uint32))]
        assert _hull_digest(hulls) == _hull_digest(hulls)

    def test_different_hull_count(self):
        v, f = _box_arrays()
        idx = f[:, :3].ravel().astype(np.uint32)
        assert _hull_digest([(v, idx)]) != _hull_digest([(v, idx), (v, idx)])

    def test_format_prefix(self):
        v, f = _box_arrays()
        idx = f[:, :3].ravel().astype(np.uint32)
        assert _hull_digest([(v, idx)]).startswith("sha256:")


class TestTraceRecorder:
    def test_record_stage(self):
        rec = TraceRecorder(config_dict={"concavity": 0.05})
        v, f = _box_arrays()
        v2 = v * 0.5
        rec.record_stage("normalize", v, v2, f, f)
        assert len(rec.events) == 1
        assert rec.events[0].stage == "normalize"
        assert rec.events[0].input_digest != rec.events[0].output_digest

    def test_record_decompose(self):
        rec = TraceRecorder()
        v, f = _box_arrays()
        idx = f.ravel().astype(np.uint32)
        hulls = [(v[:4], idx[:12]), (v[4:], idx[:12])]
        rec.record_decompose("decompose", v, f, hulls)
        assert len(rec.events) == 1
        assert rec.events[0].metadata["hull_count"] == 2

    def test_to_dict_schema_version(self):
        rec = TraceRecorder()
        d = rec.to_dict()
        assert d["trace_schema_version"] == "1.0.0"
        assert isinstance(d["events"], list)

    def test_no_timestamps(self):
        rec = TraceRecorder()
        v, f = _box_arrays()
        rec.record_stage("test", v, v, f, f)
        d = rec.to_dict()
        event_json = json.dumps(d)
        assert "timestamp" not in event_json
        assert "time" not in event_json

    def test_save_and_load(self, tmp_path):
        rec = TraceRecorder(config_dict={"concavity": 0.05})
        v, f = _box_arrays()
        rec.record_stage("normalize", v, v * 0.5, f, f)
        rec.save(tmp_path / "trace_out")

        loaded = TraceRecorder.load(tmp_path / "trace_out")
        assert len(loaded.events) == 1
        assert loaded.events[0].stage == "normalize"
        assert loaded.events[0].input_digest == rec.events[0].input_digest
        assert loaded.events[0].output_digest == rec.events[0].output_digest
        assert loaded.config_dict == {"concavity": 0.05}

    def test_save_creates_directory(self, tmp_path):
        rec = TraceRecorder()
        v, _ = _box_arrays()
        rec.record_stage("test", v, v)
        path = rec.save(tmp_path / "nested" / "trace")
        assert path.exists()

    def test_blobs_saved(self, tmp_path):
        rec = TraceRecorder()
        v, f = _box_arrays()
        rec.record_stage("test", v, v * 2, f, f)
        rec.save(tmp_path / "trace_out")
        assert (tmp_path / "trace_out" / "geometry.npz").exists()
        loaded = TraceRecorder.load(tmp_path / "trace_out")
        assert len(loaded._blobs) > 0

    def test_empty_trace_no_blobs(self, tmp_path):
        rec = TraceRecorder()
        rec.save(tmp_path / "empty_trace")
        assert (tmp_path / "empty_trace" / "trace.json").exists()
        assert not (tmp_path / "empty_trace" / "geometry.npz").exists()

    def test_multiple_events_ordered(self):
        rec = TraceRecorder()
        v, f = _box_arrays()
        rec.record_stage("normalize", v, v * 0.5, f, f)
        rec.record_stage("decimate", v * 0.5, v[:4] * 0.5, f, f[:4])
        assert [e.stage for e in rec.events] == ["normalize", "decimate"]


class TestReplay:
    def test_identical_traces(self):
        from chitin.trace_replay import replay_and_diff

        v, f = _box_arrays()
        ref = TraceRecorder()
        ref.record_stage("normalize", v, v * 0.5, f, f)

        cand = TraceRecorder()
        cand.record_stage("normalize", v, v * 0.5, f, f)

        result = replay_and_diff(ref, cand)
        assert result.identical
        assert result.divergence is None
        assert result.stages_compared == 1

    def test_output_mismatch(self):
        from chitin.trace_replay import replay_and_diff

        v, f = _box_arrays()
        ref = TraceRecorder()
        ref.record_stage("normalize", v, v * 0.5, f, f)

        cand = TraceRecorder()
        cand.record_stage("normalize", v, v * 0.7, f, f)

        result = replay_and_diff(ref, cand)
        assert not result.identical
        assert result.divergence.stage_name == "normalize"
        assert result.divergence.kind == "output_mismatch"

    def test_input_mismatch(self):
        from chitin.trace_replay import replay_and_diff

        v, f = _box_arrays()
        ref = TraceRecorder()
        ref.record_stage("normalize", v, v * 0.5, f, f)

        cand = TraceRecorder()
        cand.record_stage("normalize", v * 2, v * 0.5, f, f)

        result = replay_and_diff(ref, cand)
        assert not result.identical
        assert result.divergence.kind == "input_mismatch"

    def test_stage_name_mismatch(self):
        from chitin.trace_replay import replay_and_diff

        v, f = _box_arrays()
        ref = TraceRecorder()
        ref.record_stage("normalize", v, v, f, f)

        cand = TraceRecorder()
        cand.record_stage("decimate", v, v, f, f)

        result = replay_and_diff(ref, cand)
        assert not result.identical
        assert result.divergence.kind == "stage_name_mismatch"

    def test_extra_stage(self):
        from chitin.trace_replay import replay_and_diff

        v, f = _box_arrays()
        ref = TraceRecorder()
        ref.record_stage("normalize", v, v, f, f)
        ref.record_stage("decimate", v, v[:4], f, f[:2])

        cand = TraceRecorder()
        cand.record_stage("normalize", v, v, f, f)

        result = replay_and_diff(ref, cand)
        assert not result.identical
        assert result.divergence.kind == "extra_stage"

    def test_divergence_at_second_stage(self):
        from chitin.trace_replay import replay_and_diff

        v, f = _box_arrays()
        ref = TraceRecorder()
        ref.record_stage("normalize", v, v, f, f)
        ref.record_stage("decompose_input", v, v * 0.5, f, f)

        cand = TraceRecorder()
        cand.record_stage("normalize", v, v, f, f)
        cand.record_stage("decompose_input", v, v * 0.7, f, f)

        result = replay_and_diff(ref, cand)
        assert not result.identical
        assert result.divergence.stage_index == 1
        assert result.divergence.stage_name == "decompose_input"
        assert result.stages_compared == 1

    def test_save_load_roundtrip_replay(self, tmp_path):
        from chitin.trace_replay import replay_and_diff

        v, f = _box_arrays()
        ref = TraceRecorder()
        ref.record_stage("normalize", v, v * 0.5, f, f)
        ref.save(tmp_path / "ref")

        loaded = TraceRecorder.load(tmp_path / "ref")
        result = replay_and_diff(ref, loaded)
        assert result.identical
