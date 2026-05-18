# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

from pathlib import Path

import coacd
import numpy as np
import open3d as o3d
import trimesh

from chitin.config import Config
from chitin.plan import BuildPlan
from chitin.result import BoneInfo, ExtractionResult, Hull


def extract(
    path: str | Path,
    config: Config | None = None,
) -> ExtractionResult:
    path = Path(path)
    config = config or Config()
    suffix = path.suffix.lower()

    if suffix == ".ply":
        return _extract_from_ply(path, config)
    if suffix in (".obj", ".stl", ".off"):
        plan = BuildPlan(input_kind=suffix.lstrip("."))
        plan.step("parse")
        mesh = trimesh.load(str(path), force="mesh")
        mesh.visual = trimesh.visual.ColorVisuals()
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        plan.collider_kind = "static"
        plan.source_vertices = len(verts)
        del mesh
        return extract_from_mesh(verts, faces, config=config, _plan=plan)
    if suffix in (".glb", ".gltf", ".fbx"):
        return _extract_from_skinned_or_static(path, config)
    if suffix in (".usd", ".usda", ".usdc"):
        return _extract_from_usd(path, config)

    raise ValueError(f"Unsupported input format: {suffix}")


def extract_from_arrays(
    positions: np.ndarray,
    opacity: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    config: Config | None = None,
    _plan: BuildPlan | None = None,
) -> ExtractionResult:
    config = config or Config()
    positions = np.asarray(positions, dtype=np.float64)
    source_count = len(positions)

    if _plan is None:
        _plan = BuildPlan(input_kind="arrays", collider_kind="point_cloud")
    _plan.source_vertices = source_count

    if opacity is not None:
        raw = np.asarray(opacity, dtype=np.float64).ravel()
        if config.opacity_is_logit:
            activated = 1.0 / (1.0 + np.exp(-raw))
        else:
            activated = raw
        mask = activated >= config.opacity_threshold
        positions = positions[mask]
        if normals is not None:
            normals = np.asarray(normals, dtype=np.float64)[mask]
        _plan.step("opacity_filter")
        _plan.detected["filtered_vertices"] = int(mask.sum())

    if len(positions) < 100:
        return ExtractionResult(
            hulls=[],
            source_vertex_count=source_count,
            mesh_vertex_count=0,
            build_plan=_plan,
        )

    has_valid_normals = normals is not None and not np.allclose(normals, 0)
    if not has_valid_normals:
        _plan.step("normal_estimation")
    _plan.step("reconstruct")

    mesh = _poisson_reconstruct(positions, normals, config)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)

    if len(triangles) < 4:
        return ExtractionResult(
            hulls=[],
            source_vertex_count=source_count,
            mesh_vertex_count=len(vertices),
            build_plan=_plan,
        )

    result = _decompose_and_build(
        vertices, triangles, source_count, len(vertices), config, _plan=_plan
    )
    return result


def extract_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: Config | None = None,
    _plan: BuildPlan | None = None,
) -> ExtractionResult:
    config = config or Config()
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if _plan is None:
        _plan = BuildPlan(input_kind="mesh", collider_kind="static")
    _plan.source_vertices = _plan.source_vertices or len(vertices)
    if len(vertices) < 4 or len(faces) < 4:
        return ExtractionResult(
            hulls=[],
            source_vertex_count=len(vertices),
            mesh_vertex_count=len(vertices),
            build_plan=_plan,
        )
    return _decompose_and_build(
        vertices, faces, len(vertices), len(vertices), config, _plan=_plan
    )


def extract_from_rigged_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    joint_indices: np.ndarray,
    joint_weights: np.ndarray,
    bone_names: list[str] | None = None,
    inverse_bind_matrices: dict[int, np.ndarray] | None = None,
    config: Config | None = None,
    _plan: BuildPlan | None = None,
) -> ExtractionResult:
    config = config or Config()
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    joint_indices = np.asarray(joint_indices, dtype=np.int32)
    joint_weights = np.asarray(joint_weights, dtype=np.float64)

    if _plan is None:
        _plan = BuildPlan(input_kind="mesh", collider_kind="rigged")
    _plan.collider_kind = "rigged"
    _plan.source_vertices = len(vertices)
    _plan.detected["bone_count"] = (
        len(bone_names) if bone_names else len(np.unique(joint_indices))
    )

    source_count = len(vertices)
    _plan.step("segment_by_bone")
    segments = _segment_by_bone(vertices, faces, joint_indices, joint_weights)
    _plan.detected["segment_count"] = len(segments)

    all_hulls = []
    total_mesh_verts = 0
    bones_skipped = 0
    for bone_idx, (seg_verts, seg_faces) in segments.items():
        if len(seg_verts) < config.min_hull_vertices:
            bones_skipped += 1
            continue

        if inverse_bind_matrices and bone_idx in inverse_bind_matrices:
            ibm = inverse_bind_matrices[bone_idx]
            ones = np.ones((len(seg_verts), 1), dtype=np.float64)
            seg_verts = (np.hstack([seg_verts, ones]) @ ibm)[:, :3]

        total_mesh_verts += len(seg_verts)
        name = (
            bone_names[bone_idx]
            if bone_names and bone_idx < len(bone_names)
            else f"bone_{bone_idx}"
        )
        result = _decompose_and_build(
            seg_verts, seg_faces, len(seg_verts), len(seg_verts), config
        )
        for hull in result.hulls:
            hull.bone_name = name
            hull.bone_index = bone_idx
            all_hulls.append(hull)

    _plan.step("per_bone_decompose")
    _plan.detected["bones_skipped"] = bones_skipped
    _plan.processed_vertices = total_mesh_verts

    bones = None
    if bone_names:
        bones = []
        for idx, name in enumerate(bone_names):
            if inverse_bind_matrices and idx in inverse_bind_matrices:
                bind_xform = np.linalg.inv(inverse_bind_matrices[idx])
            else:
                bind_xform = np.eye(4, dtype=np.float64)
            bones.append(BoneInfo(name=name, index=idx, bind_transform=bind_xform))

    return ExtractionResult(
        hulls=all_hulls,
        source_vertex_count=source_count,
        mesh_vertex_count=total_mesh_verts,
        bones=bones,
        build_plan=_plan,
    )


def _extract_from_skinned_or_static(path: Path, config: Config) -> ExtractionResult:
    from chitin.gltf_skin import parse_skin

    suffix = path.suffix.lower().lstrip(".")
    plan = BuildPlan(input_kind=suffix)
    plan.step("parse")
    plan.step("skin_detect")

    skin_data = parse_skin(path)
    loaded = trimesh.load(str(path))

    has_skin_weights = (
        skin_data is not None
        and skin_data.joint_indices is not None
        and skin_data.joint_weights is not None
    )
    if has_skin_weights and isinstance(loaded, trimesh.Scene):
        plan.detected["is_skinned"] = True
        plan.detected["bone_count"] = len(skin_data.joint_names)
        mesh = loaded.to_geometry()
        if isinstance(mesh, trimesh.Trimesh):
            mesh.visual = trimesh.visual.ColorVisuals()
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            return extract_from_rigged_mesh(
                verts,
                faces,
                np.asarray(skin_data.joint_indices, dtype=np.int32),
                np.asarray(skin_data.joint_weights, dtype=np.float64),
                bone_names=skin_data.joint_names,
                inverse_bind_matrices=skin_data.inverse_bind_matrices,
                config=config,
                _plan=plan,
            )

    plan.detected["is_skinned"] = False
    plan.collider_kind = "static"

    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry()
        if isinstance(mesh, trimesh.Trimesh):
            mesh.visual = trimesh.visual.ColorVisuals()
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            del mesh
            return extract_from_mesh(verts, faces, config=config, _plan=plan)
        return ExtractionResult(
            hulls=[], source_vertex_count=0, mesh_vertex_count=0, build_plan=plan
        )

    loaded.visual = trimesh.visual.ColorVisuals()
    verts = np.asarray(loaded.vertices, dtype=np.float32)
    faces = np.asarray(loaded.faces, dtype=np.int32)
    del loaded
    return extract_from_mesh(verts, faces, config=config, _plan=plan)


def _extract_from_ply(path: Path, config: Config) -> ExtractionResult:
    from plyfile import PlyData

    plan = BuildPlan(input_kind="ply", collider_kind="point_cloud")
    plan.step("parse")

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    positions = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(
        np.float64
    )

    opacity = None
    is_logit = False
    has_opacity = "opacity" in vertex.data.dtype.names
    if has_opacity:
        opacity = np.asarray(vertex["opacity"], dtype=np.float64)
        raw_range = opacity.max() - opacity.min()
        is_logit = raw_range > 1.0 or opacity.min() < 0.0

    has_normals = all(n in vertex.data.dtype.names for n in ("nx", "ny", "nz"))
    normals = None
    if has_normals:
        normals = np.column_stack([vertex["nx"], vertex["ny"], vertex["nz"]]).astype(
            np.float64
        )

    plan.detected["has_opacity"] = has_opacity
    plan.detected["is_logit"] = is_logit
    plan.detected["has_normals"] = has_normals

    cfg = config
    if is_logit and not config.opacity_is_logit:
        cfg = Config(**{**vars(config), "opacity_is_logit": True})
    return extract_from_arrays(positions, opacity, normals, cfg, _plan=plan)


def _extract_from_usd(path: Path, config: Config) -> ExtractionResult:
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        raise ImportError(
            "USD input requires usd-core. Install with: pip install chitin[usd]"
        )

    suffix = path.suffix.lower().lstrip(".")
    plan = BuildPlan(input_kind=suffix, collider_kind="static")
    plan.step("parse")

    stage = Usd.Stage.Open(str(path))
    time_code = Usd.TimeCode.Default()
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    mesh_count = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        raw_points = mesh.GetPointsAttr().Get(time_code)
        if raw_points is None or len(raw_points) == 0:
            continue

        mesh_count += 1
        points = np.array(raw_points, dtype=np.float64)

        xformable = UsdGeom.Xformable(prim)
        world_xform = xformable.ComputeLocalToWorldTransform(time_code)
        mat = np.array(world_xform, dtype=np.float64)
        if not np.allclose(mat, np.eye(4)):
            ones = np.ones((len(points), 1), dtype=np.float64)
            homogeneous = np.hstack([points, ones])
            points = (homogeneous @ mat)[:, :3]

        face_counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        face_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

        tris = []
        idx = 0
        for count in face_counts:
            if count == 3:
                tris.append(face_indices[idx : idx + 3] + vertex_offset)
            elif count > 3:
                for j in range(1, count - 1):
                    tris.append(
                        np.array(
                            [
                                face_indices[idx] + vertex_offset,
                                face_indices[idx + j] + vertex_offset,
                                face_indices[idx + j + 1] + vertex_offset,
                            ],
                            dtype=np.int32,
                        )
                    )
            idx += count

        all_vertices.append(points)
        if tris:
            all_faces.append(np.array(tris, dtype=np.int32))
        vertex_offset += len(points)

    plan.step("world_transform")
    plan.detected["mesh_prim_count"] = mesh_count

    if not all_vertices or not all_faces:
        return ExtractionResult(
            hulls=[], source_vertex_count=0, mesh_vertex_count=0, build_plan=plan
        )

    vertices = np.concatenate(all_vertices)
    faces = np.concatenate(all_faces)
    return extract_from_mesh(vertices, faces, config, _plan=plan)


def _segment_by_bone(
    vertices: np.ndarray,
    faces: np.ndarray,
    joint_indices: np.ndarray,
    joint_weights: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    dominant_bone = joint_indices[
        np.arange(len(joint_indices)), joint_weights.argmax(axis=1)
    ]

    face_vertex_bones = dominant_bone[faces]
    face_bone = np.zeros(len(faces), dtype=np.int32)
    for i, fb in enumerate(face_vertex_bones):
        bones, counts = np.unique(fb, return_counts=True)
        face_bone[i] = bones[counts.argmax()]

    segments = {}
    for bone_idx in np.unique(face_bone):
        bone_faces = faces[face_bone == bone_idx]
        used_verts = np.unique(bone_faces)
        remap = np.full(len(vertices), -1, dtype=np.int32)
        remap[used_verts] = np.arange(len(used_verts))
        segments[int(bone_idx)] = (vertices[used_verts], remap[bone_faces])

    return segments


def _poisson_reconstruct(
    positions: np.ndarray,
    normals: np.ndarray | None,
    config: Config,
) -> o3d.geometry.TriangleMesh:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)

    has_valid_normals = normals is not None and not np.allclose(normals, 0)
    if has_valid_normals:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=10)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=config.poisson_depth
    )

    densities = np.asarray(densities)
    if len(densities) == 0:
        return mesh

    density_threshold = np.quantile(densities, 0.1)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh


def _maybe_decimate(
    vertices: np.ndarray, faces: np.ndarray, max_vertices: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(vertices) <= max_vertices:
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


def _decompose_and_build(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_count: int,
    mesh_count: int,
    config: Config,
    _plan: BuildPlan | None = None,
) -> ExtractionResult:
    pre_decimate_count = len(vertices)
    vertices, faces = _maybe_decimate(vertices, faces, config.max_decompose_vertices)

    if _plan is not None:
        if len(vertices) < pre_decimate_count:
            _plan.decimated = True
            _plan.step("decimate")
        _plan.step("decompose")
        _plan.processed_vertices = _plan.processed_vertices or len(vertices)

    tm = trimesh.Trimesh(vertices=vertices, faces=faces)
    coacd_mesh = coacd.Mesh(tm.vertices, tm.faces)

    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=config.concavity,
        preprocess_mode=config.coacd_preprocess_mode,
        preprocess_resolution=config.coacd_preprocess_resolution,
        max_convex_hull=config.max_hulls,
    )

    hulls = []
    for verts, tris in parts:
        verts = np.asarray(verts, dtype=np.float32)
        tris = np.asarray(tris, dtype=np.uint32).ravel()
        if len(verts) >= config.min_hull_vertices:
            hulls.append(Hull(vertices=verts, indices=tris))

    return ExtractionResult(
        hulls=hulls,
        source_vertex_count=source_count,
        mesh_vertex_count=mesh_count,
        build_plan=_plan,
    )
