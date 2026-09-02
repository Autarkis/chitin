"""trace=True must not affect build output — only observe it.

The trace flag is an observability option. It must not participate in:
- Resolved algorithm configuration
- Logical build identity
- Artifact digest
- Hull bytes
"""

import dataclasses
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from chitin import Config, extract_from_mesh


def _deterministic_mesh():
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    return (
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
    )


def _hull_bytes(result) -> bytes:
    return b"".join(h.vertices.tobytes() + h.indices.tobytes() for h in result.hulls)


def _phys_digest(result) -> str:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "test.phys"
        result.to_phys(str(p))
        return hashlib.sha256(p.read_bytes()).hexdigest()


def test_trace_does_not_affect_resolved_config():
    """trace is excluded from ResolvedConfig (operational, not geometric)."""
    from chitin.analyze import analyze_arrays
    from chitin.resolve import resolve_config

    v, f = _deterministic_mesh()
    analysis = analyze_arrays(v, f)

    r_off = resolve_config(Config(trace=False), analysis)
    r_on = resolve_config(Config(trace=True), analysis)

    assert dataclasses.asdict(r_off) == dataclasses.asdict(r_on)


def test_trace_does_not_affect_hull_bytes():
    """trace=True produces byte-identical hulls."""
    v, f = _deterministic_mesh()

    result_off = extract_from_mesh(v, f, config=Config(trace=False))
    result_on = extract_from_mesh(v, f, config=Config(trace=True))

    assert len(result_off.hulls) == len(result_on.hulls)
    assert _hull_bytes(result_off) == _hull_bytes(result_on)


def test_trace_does_not_affect_build_plan():
    """Build plan signals are identical with and without trace."""
    v, f = _deterministic_mesh()

    result_off = extract_from_mesh(v, f, config=Config(trace=False))
    result_on = extract_from_mesh(v, f, config=Config(trace=True))

    plan_off = result_off.build_plan
    plan_on = result_on.build_plan
    assert plan_off is not None and plan_on is not None
    assert plan_off.detected == plan_on.detected


def test_trace_does_not_affect_phys_artifact():
    """trace=True produces byte-identical .phys output."""
    v, f = _deterministic_mesh()

    result_off = extract_from_mesh(v, f, config=Config(trace=False))
    result_on = extract_from_mesh(v, f, config=Config(trace=True))

    assert _phys_digest(result_off) == _phys_digest(result_on)
