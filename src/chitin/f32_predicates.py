from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from chitin.f32_policy import QuantizationPolicy

_HULL_VOLUME_REL_TOL = 1e-3
_OUTWARD_TOL = 1e-9


@dataclass
class PlaneClassification:
    signs: np.ndarray
    positive_count: int
    negative_count: int
    on_plane_count: int


@dataclass
class ClipResult:
    vertices: np.ndarray
    faces: np.ndarray
    intersection_points: np.ndarray
    boundary_edges: np.ndarray


@dataclass
class CapResult:
    loops: list[np.ndarray]
    cap_faces: np.ndarray
    winding_consistent: bool


@dataclass
class HullResult:
    vertices: np.ndarray
    faces: np.ndarray
    face_normals: np.ndarray
    outward_consistent: bool
    volume: float


@dataclass
class PredicateDiff:
    predicate_name: str
    agrees: bool
    first_divergence: str | None
    details: dict


def _count_signs(signs: np.ndarray) -> tuple[int, int, int]:
    positive_count = int(np.sum(signs > 0))
    negative_count = int(np.sum(signs < 0))
    on_plane_count = int(np.sum(signs == 0))
    return positive_count, negative_count, on_plane_count


def classify_plane_f64(
    vertices: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray
) -> PlaneClassification:
    signed = np.dot(vertices - plane_point, plane_normal)
    signs = np.sign(signed).astype(np.int8)
    positive_count, negative_count, on_plane_count = _count_signs(signs)
    return PlaneClassification(signs, positive_count, negative_count, on_plane_count)


def _to_grid_frame(
    vertices: np.ndarray, plane_point: np.ndarray, policy: QuantizationPolicy
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # concatenate so vertices and plane_point share one centroid/scale
    combined = np.concatenate([vertices, plane_point.reshape(1, -1)], axis=0)
    grid_coords, centroid, scale_factor = policy.normalize_to_grid(combined)
    grid_vertices = grid_coords[:-1]
    grid_plane_point = grid_coords[-1]
    return grid_vertices, grid_plane_point, centroid, scale_factor


def classify_plane_f32(
    vertices: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    policy: QuantizationPolicy,
) -> PlaneClassification:
    grid_vertices, grid_plane_point, _centroid, scale_factor = _to_grid_frame(
        vertices, plane_point, policy
    )
    grid_normal_f32 = (plane_normal * scale_factor).astype(np.float32)
    dot = np.sum(
        (grid_vertices.astype(np.float32) - grid_plane_point.astype(np.float32))
        * grid_normal_f32,
        axis=1,
    )
    signs = policy.classify_sign(dot)
    positive_count, negative_count, on_plane_count = _count_signs(signs)
    return PlaneClassification(signs, positive_count, negative_count, on_plane_count)


def _clip_edge_intersection(
    v0: np.ndarray, v1: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray
) -> np.ndarray:
    denom = np.dot(v1 - v0, plane_normal)
    t = np.dot(plane_point - v0, plane_normal) / denom
    return v0 + t * (v1 - v0)


def _clip_mesh_generic(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    classify_fn,
) -> ClipResult:
    classification = classify_fn(vertices, plane_point, plane_normal)
    signs = classification.signs

    out_vertices = [vertices[i] for i in range(len(vertices))]
    out_faces: list[list[int]] = []
    new_points: list[np.ndarray] = []
    boundary_edges: list[list[int]] = []

    for tri in faces:
        tri_signs = signs[tri]
        if np.all(tri_signs >= 0):
            out_faces.append([int(tri[0]), int(tri[1]), int(tri[2])])
            continue
        if np.all(tri_signs < 0):
            continue

        # mixed: walk the triangle's edges in order, keeping positive/zero
        # vertices and inserting new intersection vertices on sign-crossing edges
        poly: list[int] = []
        tri_new_indices: list[int] = []
        n = 3
        for i in range(n):
            idx_a = int(tri[i])
            idx_b = int(tri[(i + 1) % n])
            sign_a = tri_signs[i]
            sign_b = tri_signs[(i + 1) % n]
            if sign_a >= 0:
                poly.append(idx_a)
            if sign_a * sign_b < 0:
                point = _clip_edge_intersection(
                    vertices[idx_a], vertices[idx_b], plane_point, plane_normal
                )
                new_index = len(out_vertices)
                out_vertices.append(point)
                new_points.append(point)
                poly.append(new_index)
                tri_new_indices.append(new_index)

        if len(tri_new_indices) == 2:
            boundary_edges.append([tri_new_indices[0], tri_new_indices[1]])

        for i in range(1, len(poly) - 1):
            out_faces.append([poly[0], poly[i], poly[i + 1]])

    vertices_out = np.array(out_vertices, dtype=vertices.dtype)
    faces_out = (
        np.array(out_faces, dtype=np.int64)
        if out_faces
        else np.zeros((0, 3), dtype=np.int64)
    )
    intersection_points = (
        np.array(new_points, dtype=vertices.dtype)
        if new_points
        else np.zeros((0, vertices.shape[1]), dtype=vertices.dtype)
    )
    boundary_edges_out = (
        np.array(boundary_edges, dtype=np.int64)
        if boundary_edges
        else np.zeros((0, 2), dtype=np.int64)
    )
    return ClipResult(vertices_out, faces_out, intersection_points, boundary_edges_out)


def clip_mesh_f64(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> ClipResult:
    return _clip_mesh_generic(
        vertices, faces, plane_point, plane_normal, classify_plane_f64
    )


def clip_mesh_f32(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    policy: QuantizationPolicy,
) -> ClipResult:
    grid_vertices, grid_plane_point, centroid, scale_factor = _to_grid_frame(
        vertices, plane_point, policy
    )
    grid_normal_f32 = (plane_normal * scale_factor).astype(np.float32)
    grid_vertices_f32 = grid_vertices.astype(np.float32)
    grid_plane_point_f32 = grid_plane_point.astype(np.float32)

    def classify_fn(_vertices, _plane_point, _plane_normal):
        dot = np.sum(
            (grid_vertices_f32 - grid_plane_point_f32) * grid_normal_f32, axis=1
        )
        signs = policy.classify_sign(dot)
        positive_count, negative_count, on_plane_count = _count_signs(signs)
        return PlaneClassification(
            signs, positive_count, negative_count, on_plane_count
        )

    snap_scale = 2.0 ** (policy.intersection_snap_bits - policy.grid_bits)

    def snap(point_grid: np.ndarray) -> np.ndarray:
        return np.round(point_grid * snap_scale) / snap_scale

    result_grid = _clip_mesh_generic(
        grid_vertices_f32, faces, grid_plane_point_f32, grid_normal_f32, classify_fn
    )

    n_original = len(grid_vertices_f32)
    snapped_vertices = result_grid.vertices.copy()
    snapped_vertices[n_original:] = snap(snapped_vertices[n_original:])
    snapped_intersection_points = snap(result_grid.intersection_points)

    world_vertices = (
        snapped_vertices.astype(np.float64) / policy.grid_scale / scale_factor
        + centroid
    )
    world_intersection_points = (
        snapped_intersection_points.astype(np.float64)
        / policy.grid_scale
        / scale_factor
        + centroid
    )

    return ClipResult(
        world_vertices,
        result_grid.faces,
        world_intersection_points,
        result_grid.boundary_edges,
    )


def _extract_loops(boundary_edges: np.ndarray) -> list[np.ndarray]:
    adjacency: dict[int, list[int]] = {}
    edges = [tuple(int(v) for v in edge) for edge in boundary_edges]
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    used_edges: set[frozenset] = set()
    loops: list[np.ndarray] = []
    for a, b in edges:
        key = frozenset((a, b))
        if key in used_edges:
            continue
        loop = [a, b]
        used_edges.add(key)
        current = b
        while True:
            neighbors = [
                n
                for n in adjacency.get(current, [])
                if frozenset((current, n)) not in used_edges
            ]
            if not neighbors:
                break
            nxt = neighbors[0]
            used_edges.add(frozenset((current, nxt)))
            if nxt == loop[0]:
                break
            loop.append(nxt)
            current = nxt
        loops.append(np.array(loop, dtype=np.int64))
    return loops


def _fan_triangulate_loops(loops: list[np.ndarray]) -> np.ndarray:
    faces: list[list[int]] = []
    for loop in loops:
        for i in range(1, len(loop) - 1):
            faces.append([int(loop[0]), int(loop[i]), int(loop[i + 1])])
    if not faces:
        return np.zeros((0, 3), dtype=np.int64)
    return np.array(faces, dtype=np.int64)


def _face_normal(vertices: np.ndarray, face: np.ndarray) -> np.ndarray:
    v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    return np.cross(v1 - v0, v2 - v0)


def _winding_consistent(vertices: np.ndarray, cap_faces: np.ndarray) -> bool:
    if len(cap_faces) == 0:
        return True
    first_normal = _face_normal(vertices, cap_faces[0])
    for face in cap_faces[1:]:
        normal = _face_normal(vertices, face)
        if np.dot(first_normal, normal) < 0:
            return False
    return True


def extract_cap_f64(clip_result: ClipResult) -> CapResult:
    loops = _extract_loops(clip_result.boundary_edges)
    cap_faces = _fan_triangulate_loops(loops)
    winding_consistent = _winding_consistent(clip_result.vertices, cap_faces)
    return CapResult(loops, cap_faces, winding_consistent)


def extract_cap_f32(clip_result: ClipResult, policy: QuantizationPolicy) -> CapResult:
    _ = policy
    loops = _extract_loops(clip_result.boundary_edges)
    cap_faces = _fan_triangulate_loops(loops)
    vertices_f32 = clip_result.vertices.astype(np.float32)
    winding_consistent = _winding_consistent(vertices_f32, cap_faces)
    return CapResult(loops, cap_faces, winding_consistent)


def _tetra_volume_sum(vertices: np.ndarray, faces: np.ndarray) -> float:
    total = 0.0
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        total += np.dot(v0, np.cross(v1, v2)) / 6.0
    return float(total)


def _outward_consistent(
    vertices: np.ndarray, faces: np.ndarray, face_normals: np.ndarray
) -> bool:
    hull_centroid = vertices[np.unique(faces)].mean(axis=0)
    for face, normal in zip(faces, face_normals):
        face_centroid = vertices[face].mean(axis=0)
        if np.dot(normal, face_centroid - hull_centroid) < -_OUTWARD_TOL:
            return False
    return True


def convex_hull_f64(points: np.ndarray) -> HullResult:
    hull = ConvexHull(points)
    vertices = points
    faces = hull.simplices
    face_normals = hull.equations[:, :3]
    outward_consistent = _outward_consistent(vertices, faces, face_normals)
    volume = _tetra_volume_sum(vertices, faces)
    return HullResult(vertices, faces, face_normals, outward_consistent, volume)


def convex_hull_f32(points: np.ndarray, policy: QuantizationPolicy) -> HullResult:
    points_f32 = points.astype(np.float32)
    grid_coords, _centroid, _scale_factor = policy.normalize_to_grid(points_f32)
    grid_coords_f32 = grid_coords.astype(np.float32)

    hull = ConvexHull(grid_coords_f32)
    faces = hull.simplices
    vertices = points

    points_f32_world = points.astype(np.float32)
    face_normals = hull.equations[:, :3].astype(np.float32)
    volume = _tetra_volume_sum(points_f32_world, faces)
    outward_consistent = _outward_consistent(points_f32_world, faces, face_normals)
    return HullResult(vertices, faces, face_normals, outward_consistent, volume)


def diff_classifications(
    ref: PlaneClassification, cand: PlaneClassification
) -> PredicateDiff:
    agrees = bool(np.array_equal(ref.signs, cand.signs))
    first_divergence = None
    if not agrees:
        diff_indices = np.nonzero(ref.signs != cand.signs)[0]
        i = int(diff_indices[0])
        first_divergence = (
            f"vertex {i}: ref sign {ref.signs[i]} vs cand sign {cand.signs[i]}"
        )
    else:
        diff_indices = np.array([], dtype=np.int64)
    details = {
        "ref_positive_count": ref.positive_count,
        "ref_negative_count": ref.negative_count,
        "ref_on_plane_count": ref.on_plane_count,
        "cand_positive_count": cand.positive_count,
        "cand_negative_count": cand.negative_count,
        "cand_on_plane_count": cand.on_plane_count,
        "differing_index_count": len(diff_indices),
    }
    return PredicateDiff("classify_plane", agrees, first_divergence, details)


def diff_clips(ref: ClipResult, cand: ClipResult) -> PredicateDiff:
    checks = [
        ("vertex count", len(ref.vertices), len(cand.vertices)),
        ("face count", len(ref.faces), len(cand.faces)),
        (
            "intersection point count",
            len(ref.intersection_points),
            len(cand.intersection_points),
        ),
        ("boundary edge count", len(ref.boundary_edges), len(cand.boundary_edges)),
    ]
    first_divergence = None
    for name, ref_val, cand_val in checks:
        if ref_val != cand_val:
            first_divergence = f"{name}: ref={ref_val} cand={cand_val}"
            break
    agrees = first_divergence is None
    details = {
        "ref_vertex_count": len(ref.vertices),
        "cand_vertex_count": len(cand.vertices),
        "ref_face_count": len(ref.faces),
        "cand_face_count": len(cand.faces),
        "ref_intersection_point_count": len(ref.intersection_points),
        "cand_intersection_point_count": len(cand.intersection_points),
        "ref_boundary_edge_count": len(ref.boundary_edges),
        "cand_boundary_edge_count": len(cand.boundary_edges),
    }
    return PredicateDiff("clip_mesh", agrees, first_divergence, details)


def diff_caps(ref: CapResult, cand: CapResult) -> PredicateDiff:
    first_divergence = None
    ref_loop_sizes = [len(loop) for loop in ref.loops]
    cand_loop_sizes = [len(loop) for loop in cand.loops]

    if len(ref.loops) != len(cand.loops):
        first_divergence = f"loop count: ref={len(ref.loops)} cand={len(cand.loops)}"
    elif ref_loop_sizes != cand_loop_sizes:
        first_divergence = f"loop sizes: ref={ref_loop_sizes} cand={cand_loop_sizes}"
    elif ref.winding_consistent != cand.winding_consistent:
        first_divergence = f"winding_consistent: ref={ref.winding_consistent} cand={cand.winding_consistent}"

    agrees = first_divergence is None
    details = {
        "ref_loop_count": len(ref.loops),
        "cand_loop_count": len(cand.loops),
        "ref_loop_sizes": ref_loop_sizes,
        "cand_loop_sizes": cand_loop_sizes,
        "ref_winding_consistent": ref.winding_consistent,
        "cand_winding_consistent": cand.winding_consistent,
    }
    return PredicateDiff("extract_cap", agrees, first_divergence, details)


def diff_hulls(ref: HullResult, cand: HullResult) -> PredicateDiff:
    ref_face_count = len(ref.faces)
    cand_face_count = len(cand.faces)
    rel_volume_diff = abs(ref.volume - cand.volume) / max(abs(ref.volume), 1e-30)

    first_divergence = None
    if ref_face_count != cand_face_count:
        first_divergence = f"face count: ref={ref_face_count} cand={cand_face_count}"
    elif ref.outward_consistent != cand.outward_consistent:
        first_divergence = f"outward_consistent: ref={ref.outward_consistent} cand={cand.outward_consistent}"
    elif rel_volume_diff > _HULL_VOLUME_REL_TOL:
        first_divergence = (
            f"volume relative diff: {rel_volume_diff} > {_HULL_VOLUME_REL_TOL}"
        )

    agrees = first_divergence is None
    details = {
        "ref_face_count": ref_face_count,
        "cand_face_count": cand_face_count,
        "ref_outward_consistent": ref.outward_consistent,
        "cand_outward_consistent": cand.outward_consistent,
        "ref_volume": ref.volume,
        "cand_volume": cand.volume,
        "relative_volume_diff": rel_volume_diff,
    }
    return PredicateDiff("convex_hull", agrees, first_divergence, details)
