from __future__ import annotations

import numpy as np


def outward_face_planes(hull) -> tuple[np.ndarray, np.ndarray]:
    faces = hull.indices.reshape(-1, 3)
    v = hull.vertices.astype(np.float64)
    centroid = v.mean(axis=0)
    v0 = v[faces[:, 0]]
    normals = np.cross(v[faces[:, 1]] - v0, v[faces[:, 2]] - v0)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-12, 1.0, lengths)
    normals = normals / lengths
    face_centers = (v0 + v[faces[:, 1]] + v[faces[:, 2]]) / 3.0
    outward = np.einsum("ij,ij->i", normals, face_centers - centroid)
    normals[outward < 0] *= -1
    d = np.einsum("ij,ij->i", normals, v0)
    return normals, d


def points_inside(
    normals: np.ndarray, d: np.ndarray, points: np.ndarray, tol: float = 1e-4
) -> np.ndarray:
    dots = points.astype(np.float64) @ normals.T
    return np.all(dots <= d[np.newaxis, :] + tol, axis=1)


def point_plane_margins(
    normals: np.ndarray, d: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Per-point minimum margin to the hull's face planes.

    Positive means inside by that distance; negative means outside the
    nearest violated plane by that distance.
    """
    dots = points.astype(np.float64) @ normals.T
    return (d[np.newaxis, :] - dots).min(axis=1)
