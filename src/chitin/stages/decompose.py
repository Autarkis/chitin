from __future__ import annotations

import coacd
import numpy as np
import trimesh

from chitin.plan import BuildPlan
from chitin.resolve import ResolvedConfig
from chitin.result import ExtractionResult, Hull, LodHulls
from chitin.stages.flatness import make_planar_box


def aabb_overlaps_bounds(
    hull: Hull, bounds_min: np.ndarray, bounds_max: np.ndarray
) -> bool:
    h_min = hull.vertices.min(axis=0)
    h_max = hull.vertices.max(axis=0)
    return bool(
        h_max[0] >= bounds_min[0]
        and h_min[0] <= bounds_max[0]
        and h_max[1] >= bounds_min[1]
        and h_min[1] <= bounds_max[1]
        and h_max[2] >= bounds_min[2]
        and h_min[2] <= bounds_max[2]
    )


def aabb_iou(a: Hull, b: Hull) -> float:
    a_min, a_max = a.vertices.min(axis=0), a.vertices.max(axis=0)
    b_min, b_max = b.vertices.min(axis=0), b.vertices.max(axis=0)
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter_dims = np.maximum(inter_max - inter_min, 0)
    inter_vol = float(inter_dims[0] * inter_dims[1] * inter_dims[2])
    a_vol = float(np.prod(a_max - a_min))
    b_vol = float(np.prod(b_max - b_min))
    union_vol = a_vol + b_vol - inter_vol
    if union_vol <= 0:
        return 0.0
    return inter_vol / union_vol


def dedup_overlapping_hulls(
    hulls: list[Hull], iou_threshold: float = 0.5
) -> list[Hull]:
    if len(hulls) <= 1:
        return hulls
    order = sorted(
        range(len(hulls)), key=lambda i: len(hulls[i].vertices), reverse=True
    )
    discarded: set[int] = set()
    kept: list[Hull] = []
    for pos, i in enumerate(order):
        if i in discarded:
            continue
        kept.append(hulls[i])
        for j in order[pos + 1 :]:
            if j in discarded:
                continue
            if aabb_iou(hulls[i], hulls[j]) >= iou_threshold:
                discarded.add(j)
    return kept


def maybe_decimate(
    vertices: np.ndarray, faces: np.ndarray, max_vertices: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(vertices) <= max_vertices:
        return vertices, faces
    try:
        import open3d as o3d
    except ImportError:
        return vertices, faces
    ratio = max_vertices / len(vertices)
    target_faces = max(4, int(len(faces) * ratio))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    return (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.triangles, dtype=np.int32),
    )


def extract_walkable_hulls(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[list[Hull], np.ndarray]:
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(face_normals, axis=1, keepdims=True)
    areas = np.where(areas == 0, 1e-12, areas)
    face_normals /= areas

    hist = np.abs(face_normals).sum(axis=0)
    up_axis_idx = np.argmax(hist)
    up_axis = np.zeros(3)
    up_axis[up_axis_idx] = 1.0
    if (face_normals @ up_axis).sum() < 0:
        up_axis = -up_axis

    dot = face_normals @ up_axis
    walkable_mask = dot >= np.cos(np.radians(35.0))

    if not np.any(walkable_mask):
        return [], faces

    walkable_faces = faces[walkable_mask]
    unwalkable_faces = faces[~walkable_mask]

    tm = trimesh.Trimesh(vertices=vertices, faces=walkable_faces, process=False)
    try:
        components = trimesh.graph.connected_components(tm.face_adjacency)
    except Exception:
        return [], faces

    hulls = []
    keep_unwalkable = [unwalkable_faces]

    for comp in components:
        if len(comp) < 200:
            keep_unwalkable.append(walkable_faces[comp])
            continue

        comp_faces = walkable_faces[comp]
        comp_verts = vertices[np.unique(comp_faces)]

        cv0, cv1, cv2 = (
            vertices[comp_faces[:, 0]],
            vertices[comp_faces[:, 1]],
            vertices[comp_faces[:, 2]],
        )
        comp_normals = np.cross(cv1 - cv0, cv2 - cv0)
        avg_normal = comp_normals.mean(axis=0)
        norm = np.linalg.norm(avg_normal)
        if norm > 0:
            avg_normal /= norm
        else:
            avg_normal = up_axis

        box_hull = make_planar_box(comp_verts, avg_normal)
        hulls.append(box_hull)

    final_unwalkable = (
        np.vstack(keep_unwalkable)
        if keep_unwalkable
        else np.array([], dtype=np.int32).reshape(0, 3)
    )
    return hulls, final_unwalkable


def decompose_and_build(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_count: int,
    mesh_count: int,
    config: ResolvedConfig,
    _plan: BuildPlan | None = None,
) -> ExtractionResult:
    pre_decimate_count = len(vertices)
    vertices, faces = maybe_decimate(vertices, faces, config.max_decompose_vertices)

    if _plan is not None:
        if len(vertices) < pre_decimate_count:
            _plan.decimated = True
            _plan.step("decimate")
        _plan.step("decompose")
        _plan.processed_vertices = _plan.processed_vertices or len(vertices)

    is_env = _plan is not None and _plan.detected.get("is_environment", False)
    env_hulls = []
    if is_env:
        env_hulls, faces = extract_walkable_hulls(vertices, faces)
        if env_hulls and _plan is not None:
            _plan.detected["walkable_hulls"] = len(env_hulls)

    if len(faces) < 4:
        return ExtractionResult(
            hulls=env_hulls,
            source_vertex_count=source_count,
            mesh_vertex_count=mesh_count,
            build_plan=_plan,
            lod_tiers=None,
        )

    tm = trimesh.Trimesh(vertices=vertices, faces=faces)
    coacd_mesh = coacd.Mesh(tm.vertices, tm.faces)

    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=config.concavity,
        preprocess_mode=config.coacd_preprocess_mode,
        preprocess_resolution=config.coacd_preprocess_resolution,
        max_convex_hull=config.max_hulls,
    )

    hulls = env_hulls.copy()
    for verts, tris in parts:
        verts = np.asarray(verts, dtype=np.float32)
        tris = np.asarray(tris, dtype=np.uint32).ravel()
        if len(verts) >= config.min_hull_vertices:
            hulls.append(Hull(vertices=verts, indices=tris))

    lod_tiers = None
    if config.lod_concavities:
        lod_tiers = []
        for lod_concavity in sorted(config.lod_concavities):
            lod_parts = coacd.run_coacd(
                coacd_mesh,
                threshold=lod_concavity,
                preprocess_mode=config.coacd_preprocess_mode,
                preprocess_resolution=config.coacd_preprocess_resolution,
                max_convex_hull=config.max_hulls,
            )
            lod_hulls = env_hulls.copy()
            for verts, tris in lod_parts:
                verts = np.asarray(verts, dtype=np.float32)
                tris = np.asarray(tris, dtype=np.uint32).ravel()
                if len(verts) >= config.min_hull_vertices:
                    lod_hulls.append(Hull(vertices=verts, indices=tris))
            lod_tiers.append(LodHulls(concavity=lod_concavity, hulls=lod_hulls))

    return ExtractionResult(
        hulls=hulls,
        source_vertex_count=source_count,
        mesh_vertex_count=mesh_count,
        build_plan=_plan,
        lod_tiers=lod_tiers,
    )
