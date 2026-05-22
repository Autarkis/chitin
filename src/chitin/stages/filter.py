from __future__ import annotations

import numpy as np

from chitin.resolve import ResolvedConfig


def proximity_filter_mesh(
    mesh_verts: np.ndarray,
    mesh_tris: np.ndarray,
    input_positions: np.ndarray,
    max_distance_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise ImportError(
            "Proximity filtering requires scipy. "
            "Install with: pip install chitin[splat]"
        )

    input_tree = cKDTree(input_positions)
    sample_dists, _ = input_tree.query(
        input_positions[: min(1000, len(input_positions))], k=2
    )
    median_nn = np.median(sample_dists[:, 1])

    threshold = max_distance_ratio * median_nn
    mesh_dists, _ = input_tree.query(mesh_verts)
    keep_mask = mesh_dists <= threshold

    old_to_new = np.full(len(mesh_verts), -1, dtype=np.int64)
    new_indices = np.where(keep_mask)[0]
    old_to_new[new_indices] = np.arange(len(new_indices))

    new_verts = mesh_verts[keep_mask]

    tri_keep = np.all(keep_mask[mesh_tris], axis=1)
    new_tris = old_to_new[mesh_tris[tri_keep]]

    return new_verts, new_tris.astype(np.int32)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    vnormals = np.zeros_like(vertices)
    np.add.at(vnormals, faces[:, 0], face_normals)
    np.add.at(vnormals, faces[:, 1], face_normals)
    np.add.at(vnormals, faces[:, 2], face_normals)
    norms = np.linalg.norm(vnormals, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vnormals / norms


def extrude_thin_shell(
    vertices: np.ndarray,
    faces: np.ndarray,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray]:
    vnormals = vertex_normals(vertices, faces)

    n = len(vertices)
    half = thickness / 2.0
    outer_verts = vertices + vnormals * half
    inner_verts = vertices - vnormals * half
    all_verts = np.vstack([outer_verts, inner_verts])

    outer_faces = faces.copy()
    inner_faces = faces[:, ::-1] + n

    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges_sorted = np.sort(edges, axis=1)
    edge_keys = edges_sorted[:, 0].astype(np.int64) * (n + 1) + edges_sorted[:, 1]
    unique_keys, counts = np.unique(edge_keys, return_counts=True)
    boundary_keys = set(unique_keys[counts == 1].tolist())

    stitch_faces = []
    for e0, e1 in edges:
        key = min(e0, e1) * (n + 1) + max(e0, e1)
        if key in boundary_keys:
            stitch_faces.append([e0, e1, e1 + n])
            stitch_faces.append([e0, e1 + n, e0 + n])

    if stitch_faces:
        stitch = np.array(stitch_faces, dtype=np.int32)
        all_faces = np.vstack([outer_faces, inner_faces, stitch])
    else:
        all_faces = np.vstack([outer_faces, inner_faces])

    return all_verts, all_faces


def post_poisson_filter(
    mesh_verts: np.ndarray,
    mesh_tris: np.ndarray,
    input_positions: np.ndarray,
    config: ResolvedConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if config.surface_proximity_filter > 0:
        mesh_verts, mesh_tris = proximity_filter_mesh(
            mesh_verts, mesh_tris, input_positions, config.surface_proximity_filter
        )
    if config.thin_shell and len(mesh_tris) >= 4:
        thickness = config.thin_shell_thickness
        if thickness <= 0:
            extent = mesh_verts.max(axis=0) - mesh_verts.min(axis=0)
            thickness = np.median(extent) * 0.02
        mesh_verts, mesh_tris = extrude_thin_shell(mesh_verts, mesh_tris, thickness)
    return mesh_verts, mesh_tris
