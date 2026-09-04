from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, KDTree, QhullError

from chitin.f32_policy import QuantizationPolicy

_HULL_VOLUME_REL_TOL = 1e-3
_POSITION_REL_TOL = 1e-3
_GRID_CELL_SAFETY_FACTOR = 64.0


@dataclass
class PlaneClassification:
    signs: np.ndarray
    positive_count: int
    negative_count: int
    on_plane_count: int
    fast_path_count: int = 0
    ambiguity_path_count: int = 0
    signed_distances: np.ndarray | None = None


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


def _grid_quantization_bound(grid_normal_f32: np.ndarray) -> float:
    """Max dot-product error from ±0.5 grid-cell quantization of vertex and plane point."""
    return float(np.sum(np.abs(grid_normal_f32)))


def _classify_with_fallback(
    grid_dot: np.ndarray,
    grid_normal_f32: np.ndarray,
    world_vertices: np.ndarray,
    world_plane_point: np.ndarray,
    world_plane_normal: np.ndarray,
    policy: QuantizationPolicy,
) -> tuple[np.ndarray, int, int, np.ndarray | None]:
    """Grid classification with unquantized-f32 fallback for ambiguous vertices.

    Returns (signs, fast_path_count, ambiguity_path_count, signed_distances).
    signed_distances is the array of world-frame f32 dot products when the
    ambiguity path is active, None otherwise.
    """
    signs = policy.classify_sign(grid_dot)
    n = len(signs)

    if not policy.ambiguity_fallback:
        return signs, n, 0, None

    bound = _grid_quantization_bound(grid_normal_f32)
    ambiguous = np.abs(grid_dot) <= bound
    amb_count = int(np.sum(ambiguous))

    v32 = world_vertices.astype(np.float32)
    p32 = world_plane_point.astype(np.float32)
    n32 = world_plane_normal.astype(np.float32)
    world_dot = np.sum((v32 - p32) * n32, axis=1)

    if amb_count > 0:
        amb_idx = np.nonzero(ambiguous)[0]
        signs[amb_idx] = np.sign(world_dot[amb_idx]).astype(np.int8)

    return signs, n - amb_count, amb_count, world_dot


def canonicalize_inputs_f32(
    vertices: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        vertices.astype(np.float32).astype(np.float64),
        plane_point.astype(np.float32).astype(np.float64),
        plane_normal.astype(np.float32).astype(np.float64),
    )


def categorize_disagreements(
    raw_ref_signs: np.ndarray,
    canon_ref_signs: np.ndarray,
    canon_cand_signs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex precision-loss and genuine-arithmetic boolean arrays."""
    precision_loss = raw_ref_signs != canon_ref_signs
    genuine_arithmetic = canon_ref_signs != canon_cand_signs
    return precision_loss, genuine_arithmetic


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
    if policy.canonical_f32_inputs:
        vertices, plane_point, plane_normal = canonicalize_inputs_f32(
            vertices, plane_point, plane_normal
        )
    grid_vertices, grid_plane_point, _centroid, scale_factor = _to_grid_frame(
        vertices, plane_point, policy
    )
    grid_normal_f32 = (plane_normal * scale_factor).astype(np.float32)
    dot = np.sum(
        (grid_vertices.astype(np.float32) - grid_plane_point.astype(np.float32))
        * grid_normal_f32,
        axis=1,
    )
    signs, fast_count, amb_count, signed_distances = _classify_with_fallback(
        dot, grid_normal_f32, vertices, plane_point, plane_normal, policy
    )
    positive_count, negative_count, on_plane_count = _count_signs(signs)
    return PlaneClassification(
        signs,
        positive_count,
        negative_count,
        on_plane_count,
        fast_path_count=fast_count,
        ambiguity_path_count=amb_count,
        signed_distances=signed_distances,
    )


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
    distances = classification.signed_distances

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
                if distances is not None:
                    d_a = float(distances[idx_a])
                    d_b = float(distances[idx_b])
                    t = d_a / (d_a - d_b)
                    point = vertices[idx_a] + t * (vertices[idx_b] - vertices[idx_a])
                else:
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
    if policy.canonical_f32_inputs:
        vertices, plane_point, plane_normal = canonicalize_inputs_f32(
            vertices, plane_point, plane_normal
        )
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
        signs, fast_count, amb_count, signed_distances = _classify_with_fallback(
            dot, grid_normal_f32, vertices, plane_point, plane_normal, policy
        )
        positive_count, negative_count, on_plane_count = _count_signs(signs)
        return PlaneClassification(
            signs,
            positive_count,
            negative_count,
            on_plane_count,
            fast_path_count=fast_count,
            ambiguity_path_count=amb_count,
            signed_distances=signed_distances,
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


def _outward_consistent(
    vertices: np.ndarray, faces: np.ndarray, face_normals: np.ndarray
) -> bool:
    hull_centroid = vertices[np.unique(faces)].mean(axis=0)
    for face, normal in zip(faces, face_normals):
        face_centroid = vertices[face].mean(axis=0)
        if np.dot(normal, face_centroid - hull_centroid) < 0:
            return False
    return True


def _empty_hull_result(points: np.ndarray) -> HullResult:
    return HullResult(
        points,
        np.empty((0, 3), dtype=np.int64),
        np.empty((0, 3), dtype=points.dtype),
        False,
        0.0,
    )


def convex_hull_f64(points: np.ndarray) -> HullResult:
    try:
        hull = ConvexHull(points)
    except QhullError:
        return _empty_hull_result(points)
    vertices = points
    faces = hull.simplices
    face_normals = hull.equations[:, :3]
    outward_consistent = _outward_consistent(vertices, faces, face_normals)
    volume = float(hull.volume)
    return HullResult(vertices, faces, face_normals, outward_consistent, volume)


def convex_hull_f32(points: np.ndarray, policy: QuantizationPolicy) -> HullResult:
    points_f32 = points.astype(np.float32)
    grid_coords, _centroid, _scale_factor = policy.normalize_to_grid(points_f32)
    grid_coords_f32 = grid_coords.astype(np.float32)

    try:
        hull = ConvexHull(grid_coords_f32)
    except QhullError:
        return _empty_hull_result(points)
    faces = hull.simplices
    vertices = points

    points_f32_world = points.astype(np.float32)
    face_normals = hull.equations[:, :3].astype(np.float32)
    outward_consistent = _outward_consistent(points_f32_world, faces, face_normals)
    try:
        world_hull = ConvexHull(points_f32_world)
        volume = float(world_hull.volume)
    except QhullError:
        volume = 0.0
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


def _canonical_face(face: np.ndarray) -> tuple[int, int, int]:
    a, b, c = int(face[0]), int(face[1]), int(face[2])
    return tuple(sorted((a, b, c)))


def _canonical_face_list(faces: np.ndarray) -> list[tuple[int, int, int]]:
    return sorted(_canonical_face(face) for face in faces)


def _face_set_diff(
    ref_faces: np.ndarray, cand_faces: np.ndarray
) -> tuple[bool, str | None, list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    ref_canonical = _canonical_face_list(ref_faces)
    cand_canonical = _canonical_face_list(cand_faces)
    if ref_canonical == cand_canonical:
        return True, None, ref_canonical, cand_canonical
    ref_set = set(ref_canonical)
    cand_set = set(cand_canonical)
    only_ref = sorted(ref_set - cand_set)
    only_cand = sorted(cand_set - ref_set)
    if only_ref or only_cand:
        return (
            False,
            (
                f"face set: {len(only_ref)} faces only in ref, {len(only_cand)} only in cand"
                f" (first ref-only={only_ref[0] if only_ref else None},"
                f" first cand-only={only_cand[0] if only_cand else None})"
            ),
            ref_canonical,
            cand_canonical,
        )
    return (
        False,
        f"face multiplicity differs: ref={ref_canonical} cand={cand_canonical}",
        ref_canonical,
        cand_canonical,
    )


def _characteristic_scale(*point_arrays: np.ndarray) -> float:
    pts = [p for p in point_arrays if len(p) > 0]
    if not pts:
        return 1.0
    stacked = np.concatenate(pts, axis=0)
    spread = float(np.max(np.ptp(stacked, axis=0))) if len(stacked) > 1 else 0.0
    return max(spread, 1.0)


def _match_points_by_position(
    ref_points: np.ndarray, cand_points: np.ndarray, tol: float
) -> tuple[bool, str | None, float]:
    if len(ref_points) != len(cand_points):
        return (
            False,
            f"intersection point count: ref={len(ref_points)} cand={len(cand_points)}",
            float("inf"),
        )
    if len(ref_points) == 0:
        return True, None, 0.0

    n_cand = len(cand_points)
    tree = KDTree(cand_points)
    nn_dist, nn_idx = tree.query(ref_points, k=1)
    nn_dist = np.atleast_1d(nn_dist)
    nn_idx = np.atleast_1d(nn_idx)

    order = np.argsort(nn_dist, kind="stable")
    claimed: set[int] = set()
    max_residual = 0.0
    failures: dict[int, tuple[float, str | None]] = {}

    for pos in order:
        ref_i = int(pos)
        cand_i = int(nn_idx[ref_i])
        best_dist = float(nn_dist[ref_i])
        if cand_i in claimed:
            k = 2
            while True:
                k = min(k, n_cand)
                q_dist, q_idx = tree.query(ref_points[ref_i], k=k)
                q_dist = np.atleast_1d(q_dist)
                q_idx = np.atleast_1d(q_idx)
                found = False
                for d, idx in zip(q_dist, q_idx):
                    idx = int(idx)
                    if idx not in claimed:
                        best_dist = float(d)
                        cand_i = idx
                        found = True
                        break
                if found or k >= n_cand:
                    break
                k *= 2
        claimed.add(cand_i)
        max_residual = max(max_residual, best_dist)
        if best_dist > tol:
            failures[ref_i] = (
                best_dist,
                f"intersection point {ref_i}: nearest cand match distance {best_dist} > tol {tol}",
            )

    if failures:
        first_i = min(failures)
        return False, failures[first_i][1], max_residual
    return True, None, max_residual


def diff_clips(
    ref: ClipResult, cand: ClipResult, policy: QuantizationPolicy | None = None
) -> PredicateDiff:
    faces_agree, faces_divergence, _ref_canonical, _cand_canonical = _face_set_diff(
        ref.faces, cand.faces
    )
    scale = _characteristic_scale(ref.intersection_points, cand.intersection_points)
    if policy is not None:
        tol = _GRID_CELL_SAFETY_FACTOR * scale / policy.grid_scale
    else:
        tol = _POSITION_REL_TOL * scale
    points_agree, points_divergence, max_residual = _match_points_by_position(
        ref.intersection_points, cand.intersection_points, tol
    )

    agrees = faces_agree and points_agree
    first_divergence = faces_divergence if not faces_agree else points_divergence

    details = {
        "face_set_agrees": faces_agree,
        "intersection_points_agree": points_agree,
        "intersection_max_residual": max_residual,
        "intersection_tolerance": tol,
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


def _canonicalize_loop(loop: np.ndarray) -> tuple[int, ...]:
    n = len(loop)
    if n == 0:
        return ()
    indices = [int(v) for v in loop]
    rotations = [tuple(indices[i:] + indices[:i]) for i in range(n)]
    return min(rotations)


def diff_caps(ref: CapResult, cand: CapResult) -> PredicateDiff:
    ref_loops_canonical = sorted(_canonicalize_loop(loop) for loop in ref.loops)
    cand_loops_canonical = sorted(_canonicalize_loop(loop) for loop in cand.loops)
    loops_agree = ref_loops_canonical == cand_loops_canonical

    faces_agree, faces_divergence, _ref_face_canonical, _cand_face_canonical = (
        _face_set_diff(ref.cap_faces, cand.cap_faces)
    )

    winding_agrees = ref.winding_consistent == cand.winding_consistent

    first_divergence = None
    if not loops_agree:
        first_divergence = (
            f"loop topology: ref={ref_loops_canonical} cand={cand_loops_canonical}"
        )
    elif not faces_agree:
        first_divergence = faces_divergence
    elif not winding_agrees:
        first_divergence = f"winding_consistent: ref={ref.winding_consistent} cand={cand.winding_consistent}"

    agrees = loops_agree and faces_agree and winding_agrees
    ref_loop_sizes = [len(loop) for loop in ref.loops]
    cand_loop_sizes = [len(loop) for loop in cand.loops]
    details = {
        "loops_agree": loops_agree,
        "cap_face_set_agrees": faces_agree,
        "ref_loop_count": len(ref.loops),
        "cand_loop_count": len(cand.loops),
        "ref_loop_sizes": ref_loop_sizes,
        "cand_loop_sizes": cand_loop_sizes,
        "ref_loops_canonical": ref_loops_canonical,
        "cand_loops_canonical": cand_loops_canonical,
        "ref_winding_consistent": ref.winding_consistent,
        "cand_winding_consistent": cand.winding_consistent,
    }
    return PredicateDiff("extract_cap", agrees, first_divergence, details)


def _face_normal_sign(
    vertices: np.ndarray, face: tuple[int, int, int], hull_centroid: np.ndarray
) -> int:
    v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    normal = np.cross(v1 - v0, v2 - v0)
    face_centroid = (v0 + v1 + v2) / 3.0
    dot = float(np.dot(normal, face_centroid - hull_centroid))
    return 1 if dot >= 0 else -1


def _orientation_signs(
    vertices: np.ndarray, canonical_faces: list[tuple[int, int, int]]
) -> dict[tuple[int, int, int], int]:
    if not canonical_faces:
        return {}
    used = sorted({idx for face in canonical_faces for idx in face})
    hull_centroid = vertices[used].mean(axis=0)
    return {
        face: _face_normal_sign(vertices, face, hull_centroid)
        for face in canonical_faces
    }


def diff_hulls(ref: HullResult, cand: HullResult) -> PredicateDiff:
    faces_agree, faces_divergence, ref_canonical, cand_canonical = _face_set_diff(
        ref.faces, cand.faces
    )

    shared_faces = sorted(set(ref_canonical) & set(cand_canonical))
    ref_signs = _orientation_signs(ref.vertices, shared_faces)
    cand_signs = _orientation_signs(cand.vertices, shared_faces)
    orientation_agrees = True
    orientation_divergence = None
    for face in shared_faces:
        if ref_signs[face] != cand_signs[face]:
            orientation_agrees = False
            orientation_divergence = (
                f"face normal orientation {face}: ref sign {ref_signs[face]} "
                f"vs cand sign {cand_signs[face]}"
            )
            break

    rel_volume_diff = abs(ref.volume - cand.volume) / max(abs(ref.volume), 1.0)
    volume_agrees = rel_volume_diff <= _HULL_VOLUME_REL_TOL

    first_divergence = None
    if not faces_agree:
        first_divergence = faces_divergence
    elif not orientation_agrees:
        first_divergence = orientation_divergence
    elif not volume_agrees:
        first_divergence = (
            f"volume relative diff: {rel_volume_diff} > {_HULL_VOLUME_REL_TOL}"
        )

    agrees = faces_agree and orientation_agrees and volume_agrees
    details = {
        "face_set_agrees": faces_agree,
        "orientation_agrees": orientation_agrees,
        "ref_face_count": len(ref.faces),
        "cand_face_count": len(cand.faces),
        "ref_outward_consistent": ref.outward_consistent,
        "cand_outward_consistent": cand.outward_consistent,
        "ref_volume": ref.volume,
        "cand_volume": cand.volume,
        "relative_volume_diff": rel_volume_diff,
    }
    return PredicateDiff("convex_hull", agrees, first_divergence, details)
