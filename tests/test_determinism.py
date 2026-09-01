"""Same input bytes + same config = same output bytes.

The manifest hashes, the service's output cache and any re-run of an archived
build all rest on this. It is not free: CoACD's search is OpenMP-parallel and
its thread scheduling decides which decomposition it settles on, so an unpinned
run returned a different hull count and different bytes nearly every time
(47/48/50 hulls over four runs of one 3.6k-face mesh, four distinct hashes).
"""

import numpy as np
import pytest
import trimesh

from chitin import Config, extract_from_mesh


def _l_shape():
    """A watertight concave mesh: small, but concave enough that CoACD searches."""
    mesh = trimesh.util.concatenate(
        [
            trimesh.creation.box(extents=[2, 0.5, 0.5]),
            trimesh.creation.box(
                extents=[0.5, 2, 0.5],
                transform=trimesh.transformations.translation_matrix([0.75, 0.75, 0]),
            ),
        ]
    )
    return (
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
    )


def _hull_bytes(result) -> bytes:
    return b"".join(h.vertices.tobytes() + h.indices.tobytes() for h in result.hulls)


def test_repeated_decomposition_is_byte_identical():
    verts, faces = _l_shape()
    config = Config(concavity=0.1)

    first = extract_from_mesh(verts, faces, config=config)
    second = extract_from_mesh(verts, faces, config=config)

    # A single-hull result would pass without the search ever branching, and a
    # bounding-box fallback would compare two AABBs -- neither tests anything.
    assert len(first.hulls) > 1
    assert first.build_plan.detected.get("fallback_hulls", 0) == 0
    assert first.build_plan.detected["coacd_deterministic"] is True

    assert len(first.hulls) == len(second.hulls)
    assert _hull_bytes(first) == _hull_bytes(second)


def test_timeout_raises_compilation_error():
    # The budget exists to kill a native stall. It was hardcoded at 15s, below
    # the real decomposition time of ordinary concave inputs, which used to
    # silently replace hulls with a bounding box; a timeout is a compilation
    # failure now, never a degraded artifact (chitin #102).
    from chitin.errors import CompilationError

    verts, faces = _l_shape()
    with pytest.raises(CompilationError) as exc_info:
        extract_from_mesh(
            verts, faces, config=Config(concavity=0.1, coacd_timeout=0.001)
        )

    assert exc_info.value.code == "COACD_TIMEOUT"
    assert "timeout_seconds" in exc_info.value.evidence
