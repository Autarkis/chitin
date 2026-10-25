"""Regression: on-plane cap boundary edges must be emitted exactly once.

`_clip_mesh_generic` tallies, per undirected edge, how many kept triangles
have both endpoints exactly on the clipping plane (sign 0). Odd tally means a
genuine cap boundary edge. A duplicated tally-to-boundary loop once appended
every such edge to `boundary_edges` twice; these tests pin the single emission.
"""

from __future__ import annotations

import numpy as np
import pytest

from chitin.f32_policy import POLICY_0_3_0
from chitin.f32_predicates import clip_mesh_f32, clip_mesh_f64

PLANE_POINT = np.array([0.0, 0.0, 0.0], dtype=np.float64)
PLANE_NORMAL = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _octahedron() -> tuple[np.ndarray, np.ndarray]:
    """Closed octahedron; the equator ring (v0..v3) lies exactly at z = 0."""
    vertices = np.array(
        [
            [1.0, 0.0, 0.0],  # 0  equator
            [0.0, 1.0, 0.0],  # 1  equator
            [-1.0, 0.0, 0.0],  # 2  equator
            [0.0, -1.0, 0.0],  # 3  equator
            [0.0, 0.0, 1.0],  # 4  top apex
            [0.0, 0.0, -1.0],  # 5  bottom apex
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            [1, 0, 5],
            [2, 1, 5],
            [3, 2, 5],
            [0, 3, 5],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _cube_on_plane() -> tuple[np.ndarray, np.ndarray]:
    """Closed unit cube whose bottom face lies exactly on z = 0."""
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],  # bottom (z = 0)
            [4, 5, 6],
            [4, 6, 7],  # top
            [0, 1, 5],
            [0, 5, 4],  # y = 0
            [1, 2, 6],
            [1, 6, 5],  # x = 1
            [2, 3, 7],
            [2, 7, 6],  # y = 1
            [3, 0, 4],
            [3, 4, 7],  # x = 0
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _clips(vertices: np.ndarray, faces: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "f64": clip_mesh_f64(vertices, faces, PLANE_POINT, PLANE_NORMAL),
        "f32": clip_mesh_f32(
            vertices.astype(np.float32),
            faces,
            PLANE_POINT.astype(np.float32),
            PLANE_NORMAL.astype(np.float32),
            POLICY_0_3_0,
        ),
    }


def _undirected_rows(edges: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(sorted(int(v) for v in row)) for row in edges]


@pytest.mark.parametrize("backend", ["f64", "f32"])
def test_octahedron_equator_edges_emitted_once(backend: str) -> None:
    """Plane z = 0 through the octahedron equator.

    Kept triangles (all signs >= 0): the 4 upper faces, each (+, 0, 0) with
    exactly one on-plane edge (the equator edge v_i-v_{i+1}). The 4 lower
    faces are (-, 0, 0): mixed but with no strict sign crossing, so they emit
    no faces and are not tallied. Each of the 4 equator edges is therefore
    tallied by exactly one kept triangle -> odd -> a genuine boundary edge.
    No sign-crossing edges exist, so there are no intersection-derived
    boundary edges. Expected: exactly 4 boundary edges, all distinct.
    """
    vertices, faces = _octahedron()
    edges = _clips(vertices, faces)[backend].boundary_edges
    rows = _undirected_rows(edges)

    assert len(rows) == len(set(rows)), f"duplicate boundary edges: {rows}"
    assert len(rows) == 4
    assert set(rows) == {(0, 1), (1, 2), (2, 3), (0, 3)}


@pytest.mark.parametrize("backend", ["f64", "f32"])
def test_cube_bottom_on_plane_has_no_boundary_edges(backend: str) -> None:
    """Cube resting on z = 0: every on-plane edge is interior to the kept surface.

    All 12 triangles are kept (signs >= 0). The 4 perimeter edges of the
    bottom face are each tallied by one bottom triangle and one side triangle
    -> 2 (even); the bottom diagonal v0-v2 is tallied by both bottom triangles
    -> 2 (even). No odd tallies, so expected boundary edges: 0 (and still no
    duplicates).
    """
    vertices, faces = _cube_on_plane()
    edges = _clips(vertices, faces)[backend].boundary_edges
    rows = _undirected_rows(edges)

    assert len(rows) == len(set(rows))
    assert len(rows) == 0
