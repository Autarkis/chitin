"""Poisson reconstruction isolation policy (numpy-only; open3d not required)."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import numpy as np

from chitin.stages import reconstruct as R


def _cfg(depth):
    # poisson_reconstruct only reads these two attributes.
    return SimpleNamespace(poisson_depth=depth, poisson_density_quantile=0.1)


def test_risky_poisson_depth_forces_subprocess_isolation(monkeypatch):
    calls = {"inner": 0, "sub": 0}

    def boom_inner(*a, **k):
        calls["inner"] += 1
        raise AssertionError("in-process reconstruct must not run at a risky depth")

    def fake_run(cmd, **k):
        calls["sub"] += 1
        np.savez(
            cmd[3],  # out_path
            vertices=np.zeros((3, 3), dtype=np.float64),
            triangles=np.zeros((1, 3), dtype=np.int32),
        )
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(R, "poisson_reconstruct_inner", boom_inner)
    monkeypatch.setattr(subprocess, "run", fake_run)

    R.poisson_reconstruct(np.zeros((10, 3)), None, _cfg(R.RISKY_POISSON_DEPTH))
    assert calls["sub"] == 1 and calls["inner"] == 0


def test_safe_poisson_depth_runs_in_process(monkeypatch):
    calls = {"inner": 0}

    def rec_inner(*a, **k):
        calls["inner"] += 1
        return np.zeros((3, 3)), np.zeros((1, 3), dtype=np.int32)

    monkeypatch.setattr(R, "poisson_reconstruct_inner", rec_inner)

    R.poisson_reconstruct(np.zeros((10, 3)), None, _cfg(R.RISKY_POISSON_DEPTH - 1))
    assert calls["inner"] == 1
