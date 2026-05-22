from __future__ import annotations

import numpy as np

from chitin.result import Hull


def is_flat_mesh(
    vertices: np.ndarray, faces: np.ndarray, threshold: float = 0.9
) -> tuple[bool, np.ndarray | None]:
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(face_normals, axis=1, keepdims=True)
    areas = np.where(areas == 0, 1e-12, areas)
    face_normals = face_normals / areas

    cov = (face_normals.T * areas.ravel()) @ face_normals / areas.sum()
    eigenvalues = np.linalg.eigvalsh(cov)
    dominant_ratio = eigenvalues[2] / max(eigenvalues.sum(), 1e-12)

    if dominant_ratio < threshold:
        return False, None

    dominant_normal = np.linalg.eigh(cov)[1][:, 2]
    return True, dominant_normal


def make_planar_box(vertices: np.ndarray, dominant_normal: np.ndarray) -> Hull:
    n = dominant_normal / np.linalg.norm(dominant_normal)

    abs_n = np.abs(n)
    if abs_n[0] <= abs_n[1] and abs_n[0] <= abs_n[2]:
        ref = np.array([1.0, 0.0, 0.0])
    elif abs_n[1] <= abs_n[2]:
        ref = np.array([0.0, 1.0, 0.0])
    else:
        ref = np.array([0.0, 0.0, 1.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    center = vertices.mean(axis=0)
    local = vertices - center
    proj_u = local @ u
    proj_v = local @ v
    proj_n = local @ n

    half_u = (proj_u.max() - proj_u.min()) / 2.0
    half_v = (proj_v.max() - proj_v.min()) / 2.0
    half_n = max((proj_n.max() - proj_n.min()) / 2.0, half_u * 0.02)
    center_u = (proj_u.max() + proj_u.min()) / 2.0
    center_v = (proj_v.max() + proj_v.min()) / 2.0
    center_n = (proj_n.max() + proj_n.min()) / 2.0
    center = center + u * center_u + v * center_v + n * center_n

    corners = np.array(
        [
            center + u * s_u * half_u + v * s_v * half_v + n * s_n * half_n
            for s_u in (-1, 1)
            for s_v in (-1, 1)
            for s_n in (-1, 1)
        ],
        dtype=np.float32,
    )

    indices = np.array(
        [
            0,
            2,
            6,
            0,
            6,
            4,
            1,
            5,
            7,
            1,
            7,
            3,
            0,
            1,
            3,
            0,
            3,
            2,
            4,
            6,
            7,
            4,
            7,
            5,
            0,
            4,
            5,
            0,
            5,
            1,
            2,
            3,
            7,
            2,
            7,
            6,
        ],
        dtype=np.uint32,
    )

    return Hull(vertices=corners, indices=indices)
