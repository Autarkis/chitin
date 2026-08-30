"""Hints from preflight's cheap hollow-shell heuristic.

preflight.detect_environment_hints() only ever sees the raw input path — it
runs before Config resolution, so it cannot know whether the pipeline will
later default surface_proximity_filter to 5.0 for this input (resolve.py
does that for any point-cloud reconstruction). The hint text must therefore
not tell the user to pass a flag that may already be in effect, and must not
present --thin-shell as an unqualified recommendation (measured net harmful
on the one large scene it has been benchmarked on).
"""

from __future__ import annotations

import struct

import numpy as np

from chitin.preflight import detect_environment_hints

_ROOM = (6.0, 3.0, 5.0)


def _plane(rng, n, axis, value, size, jitter=0.02):
    pts = np.empty((n, 3))
    pts[:, axis] = value + rng.normal(0.0, jitter, n)
    for other in (i for i in range(3) if i != axis):
        pts[:, other] = rng.uniform(0.0, size[other], n)
    return pts


def _hollow_shell_points(n_per_face=200):
    rng = np.random.default_rng(0)
    faces = [(axis, value) for axis in range(3) for value in (0.0, _ROOM[axis])]
    return np.vstack([_plane(rng, n_per_face, a, v, _ROOM) for a, v in faces])


def _write_point_ply(path, positions):
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(positions)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.writelines(struct.pack("<fff", x, y, z) for x, y, z in positions)


def test_hollow_shell_hint_does_not_recommend_proximity_filter(tmp_path):
    p = tmp_path / "shell.ply"
    _write_point_ply(p, _hollow_shell_points())

    hints = detect_environment_hints(p)

    assert hints is not None
    hint = hints[0]
    assert "hollow shell" in hint
    # Point-cloud input already defaults surface_proximity_filter to 5.0
    # (resolve.py); the hint must not prescribe re-applying it.
    assert "--proximity-filter" not in hint
    # --thin-shell is measured net harmful on the one scene benchmarked so
    # far, so it must read as optional, not a directive.
    assert "optional" in hint
    assert "consider --thin-shell" not in hint
    assert "defaults on" in hint
