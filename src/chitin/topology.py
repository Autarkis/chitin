"""Orthogonal mesh topology analysis.

All facts are independent and computed from raw vertex/face arrays via pure
NumPy. No trimesh, no open3d. Each fact answers one question about the input
mesh; consumers combine them for backend admission or quality gating.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TopologyAnalysis:
    """Orthogonal topology facts for one mesh (or one connected component)."""

    component_count: int
    boundary_edge_count: int
    non_manifold_edge_count: int
    degenerate_face_count: int
    consistently_oriented: bool
    signed_volume_by_component: list[float]
    closed: bool
    two_manifold: bool

    @property
    def manifold_and_closed(self) -> bool:
        return self.closed and self.two_manifold

    def admits_backend(self, backend: str) -> bool:
        """Whether this topology passes the admission gate for a given backend."""
        if backend == "webgpu":
            return (
                self.closed
                and self.two_manifold
                and self.non_manifold_edge_count == 0
                and self.degenerate_face_count == 0
            )
        return True

    def to_dict(self) -> dict:
        return {
            "component_count": self.component_count,
            "boundary_edge_count": self.boundary_edge_count,
            "non_manifold_edge_count": self.non_manifold_edge_count,
            "degenerate_face_count": self.degenerate_face_count,
            "consistently_oriented": self.consistently_oriented,
            "signed_volume_by_component": list(self.signed_volume_by_component),
            "closed": self.closed,
            "two_manifold": self.two_manifold,
        }


def _edge_array(faces: np.ndarray) -> np.ndarray:
    """All directed half-edges from the face array."""
    return np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )


def _edge_counts(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted undirected edges and their occurrence counts."""
    edges = _edge_array(faces)
    sorted_edges = np.sort(edges, axis=1)
    unique, counts = np.unique(sorted_edges, axis=0, return_counts=True)
    return unique, counts


def _count_boundary_edges(faces: np.ndarray) -> int:
    """Edges shared by exactly one triangle (open boundary)."""
    if len(faces) == 0:
        return 0
    _, counts = _edge_counts(faces)
    return int(np.sum(counts == 1))


def _count_non_manifold_edges(faces: np.ndarray) -> int:
    """Edges shared by more than two triangles."""
    if len(faces) == 0:
        return 0
    _, counts = _edge_counts(faces)
    return int(np.sum(counts > 2))


def _count_degenerate_faces(vertices: np.ndarray, faces: np.ndarray) -> int:
    """Faces with zero area (collapsed triangles)."""
    if len(faces) == 0:
        return 0
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    crosses = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(crosses, axis=1)
    return int(np.sum(areas < 1e-12))


def _check_consistent_orientation(faces: np.ndarray) -> bool:
    """Check if all adjacent faces agree on half-edge direction.

    In a consistently oriented mesh, for every interior edge, the two
    triangles sharing it traverse the edge in opposite directions.
    """
    if len(faces) == 0:
        return True
    half_edges = _edge_array(faces)
    he_tuples = [tuple(e) for e in half_edges]
    from collections import Counter

    counts = Counter(he_tuples)
    for c in counts.values():
        if c > 1:
            return False
    return True


def _connected_components(faces: np.ndarray, n_verts: int) -> list[np.ndarray]:
    """Return face index arrays, one per vertex-connected component."""
    if len(faces) == 0:
        return []
    parent = np.arange(n_verts, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    for f in faces:
        union(f[0], f[1])
        union(f[1], f[2])

    face_roots = np.array(
        [find(faces[i, 0]) for i in range(len(faces))], dtype=np.int64
    )
    unique_roots = np.unique(face_roots)
    components = []
    for r in unique_roots:
        components.append(np.where(face_roots == r)[0])
    return components


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume of a triangle mesh via the divergence theorem."""
    if len(faces) == 0:
        return 0.0
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    vol = np.sum(
        v0[:, 0] * (v1[:, 1] * v2[:, 2] - v1[:, 2] * v2[:, 1])
        + v0[:, 1] * (v1[:, 2] * v2[:, 0] - v1[:, 0] * v2[:, 2])
        + v0[:, 2] * (v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    )
    return float(vol / 6.0)


def analyze_topology(vertices: np.ndarray, faces: np.ndarray) -> TopologyAnalysis:
    """Compute all orthogonal topology facts for the given mesh."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    if len(faces) == 0:
        return TopologyAnalysis(
            component_count=0,
            boundary_edge_count=0,
            non_manifold_edge_count=0,
            degenerate_face_count=0,
            consistently_oriented=True,
            signed_volume_by_component=[],
            closed=False,
            two_manifold=True,
        )

    boundary = _count_boundary_edges(faces)
    non_manifold = _count_non_manifold_edges(faces)
    degenerate = _count_degenerate_faces(vertices, faces)
    oriented = _check_consistent_orientation(faces)

    components = _connected_components(faces, len(vertices))
    volumes = []
    for comp_faces_idx in components:
        comp_faces = faces[comp_faces_idx]
        volumes.append(_signed_volume(vertices, comp_faces))

    closed = boundary == 0 and non_manifold == 0
    two_manifold = non_manifold == 0

    return TopologyAnalysis(
        component_count=len(components),
        boundary_edge_count=boundary,
        non_manifold_edge_count=non_manifold,
        degenerate_face_count=degenerate,
        consistently_oriented=oriented,
        signed_volume_by_component=volumes,
        closed=closed,
        two_manifold=two_manifold,
    )
