"""maybe_decimate must not silently skip when Open3D is unavailable."""

from __future__ import annotations

import builtins
import logging

import numpy as np

from chitin.stages.decompose import maybe_decimate


def test_warns_when_over_threshold_and_open3d_unavailable(monkeypatch, caplog):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "open3d":
            raise ImportError("simulated: open3d not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    verts = np.zeros((10, 3), dtype=np.float64)
    faces = np.zeros((4, 3), dtype=np.int32)
    with caplog.at_level(logging.WARNING, logger="chitin"):
        out_v, out_f = maybe_decimate(verts, faces, max_vertices=5)

    # Over the threshold but Open3D absent: mesh unchanged AND a warning emitted.
    assert len(out_v) == 10
    assert any("decimation" in r.getMessage().lower() for r in caplog.records)


def test_noop_and_silent_under_threshold(caplog):
    verts = np.zeros((3, 3), dtype=np.float64)
    faces = np.zeros((1, 3), dtype=np.int32)
    with caplog.at_level(logging.WARNING, logger="chitin"):
        out_v, _ = maybe_decimate(verts, faces, max_vertices=5)

    assert len(out_v) == 3
    assert not caplog.records  # under the threshold: nothing to warn about
