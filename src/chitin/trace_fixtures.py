"""License-clean procedural fixtures for reference traces.

Each fixture is a pure-numpy mesh designed to exercise a specific
decomposition scenario. No external assets, no license concerns.
"""

from __future__ import annotations

import numpy as np


def box_mesh(
    extents: tuple[float, float, float] = (2.0, 2.0, 2.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box. Convex baseline — CoACD should return ~1 hull."""
    hx, hy, hz = [e / 2 for e in extents]
    vertices = np.array(
        [
            [-hx, -hy, -hz],
            [-hx, -hy, hz],
            [-hx, hy, -hz],
            [-hx, hy, hz],
            [hx, -hy, -hz],
            [hx, -hy, hz],
            [hx, hy, -hz],
            [hx, hy, hz],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def l_shape_mesh() -> tuple[np.ndarray, np.ndarray]:
    """L-shaped concave mesh. Forces multi-hull decomposition."""
    vertices = np.array(
        [
            # Bottom-left block
            [0, 0, 0],
            [2, 0, 0],
            [2, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 1],
            [2, 1, 1],
            [0, 1, 1],
            # Top-left block (extends upward)
            [0, 1, 0],
            [1, 1, 0],
            [1, 2, 0],
            [0, 2, 0],
            [0, 1, 1],
            [1, 1, 1],
            [1, 2, 1],
            [0, 2, 1],
        ],
        dtype=np.float32,
    )
    # Merge shared vertices (indices 3=8, 7=12)
    # Keep as separate blocks with shared-vertex faces for clarity
    faces = np.array(
        [
            # Bottom block faces
            [0, 1, 2],
            [0, 2, 3],
            [4, 7, 6],
            [4, 6, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 6, 7],
            [2, 7, 3],
            [0, 3, 7],
            [0, 7, 4],
            [1, 5, 6],
            [1, 6, 2],
            # Top block faces
            [8, 9, 10],
            [8, 10, 11],
            [12, 15, 14],
            [12, 14, 13],
            [8, 12, 13],
            [8, 13, 9],
            [10, 14, 15],
            [10, 15, 11],
            [8, 11, 15],
            [8, 15, 12],
            [9, 13, 14],
            [9, 14, 10],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def thin_panel_mesh(
    width: float = 2.0,
    height: float = 2.0,
    thickness: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Thin panel — tests near-degenerate geometry handling."""
    hw, hh, ht = width / 2, height / 2, thickness / 2
    vertices = np.array(
        [
            [-hw, -hh, -ht],
            [hw, -hh, -ht],
            [hw, hh, -ht],
            [-hw, hh, -ht],
            [-hw, -hh, ht],
            [hw, -hh, ht],
            [hw, hh, ht],
            [-hw, hh, ht],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 7, 6],
            [4, 6, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 6, 7],
            [2, 7, 3],
            [0, 3, 7],
            [0, 7, 4],
            [1, 5, 6],
            [1, 6, 2],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def disconnected_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Two separated boxes. Tests component splitting."""
    v1, f1 = box_mesh((1.0, 1.0, 1.0))
    v2, f2 = box_mesh((1.0, 1.0, 1.0))
    v2 = v2 + np.array([5.0, 0.0, 0.0], dtype=np.float32)
    vertices = np.concatenate([v1, v2])
    faces = np.concatenate([f1, f2 + len(v1)])
    return vertices, faces


def degenerate_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Box with one degenerate (zero-area) face appended."""
    v, f = box_mesh()
    degen_face = np.array([[0, 0, 1]], dtype=np.int32)
    faces = np.concatenate([f, degen_face])
    return v, faces


def high_complexity_mesh(subdivisions: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Icosphere with many faces — stress test for decomposition.

    Pure numpy icosphere (no trimesh dependency for fixture generation).
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Icosahedron base vertices
    verts = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    # Normalize to unit sphere
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    tris = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )

    # Subdivide
    for _ in range(subdivisions):
        edge_midpoints: dict[tuple[int, int], int] = {}
        new_tris = []
        verts_list = list(verts)

        for tri in tris:
            mids = []
            for j in range(3):
                a, b = int(tri[j]), int(tri[(j + 1) % 3])
                key = (min(a, b), max(a, b))
                if key not in edge_midpoints:
                    mid = (np.array(verts_list[a]) + np.array(verts_list[b])) / 2
                    mid /= np.linalg.norm(mid)
                    edge_midpoints[key] = len(verts_list)
                    verts_list.append(mid)
                mids.append(edge_midpoints[key])

            v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])
            m01, m12, m20 = mids
            new_tris.extend(
                [
                    [v0, m01, m20],
                    [v1, m12, m01],
                    [v2, m20, m12],
                    [m01, m12, m20],
                ]
            )

        verts = np.array(verts_list, dtype=np.float64)
        tris = np.array(new_tris, dtype=np.int32)

    return verts.astype(np.float32), tris


# Registry of all fixtures
FIXTURES = {
    "box": box_mesh,
    "l_shape": l_shape_mesh,
    "thin_panel": thin_panel_mesh,
    "disconnected": disconnected_mesh,
    "degenerate": degenerate_mesh,
    "high_complexity": high_complexity_mesh,
}
