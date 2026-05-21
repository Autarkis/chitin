# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

from pathlib import Path

import numpy as np

from chitin.adapters import load
from chitin.analyze import analyze_arrays
from chitin.config import Config
from chitin.plan import BuildPlan
from chitin.resolve import ResolvedConfig, resolve_config
from chitin.result import BoneInfo, ExtractionResult
from chitin.stages.decompose import decompose_and_build
from chitin.stages.filter import post_poisson_filter
from chitin.stages.reconstruct import poisson_reconstruct
from chitin.stages.segment import segment_by_bone
from chitin.stages.splat import (
    extract_spatial,
    inflate_splat_points,
    normals_from_covariance,
)


def extract(
    path: str | Path,
    config: Config | None = None,
) -> ExtractionResult:
    path = Path(path)
    config = config or Config()

    result = load(path)
    plan = BuildPlan(input_kind=result.format)
    plan.step("parse")
    plan.detected.update(result.detected)

    analysis = analyze_arrays(
        result.positions,
        result.opacity,
        result.scales,
        result.rots,
        format=result.format,
        face_count=len(result.faces) if result.faces is not None else None,
        is_skinned=result.skin is not None,
    )
    resolved = resolve_config(config, analysis)

    if result.skin is not None:
        plan.collider_kind = "rigged"
        return extract_from_rigged_mesh(
            result.positions,
            result.faces,
            result.skin.joint_indices,
            result.skin.joint_weights,
            bone_names=result.skin.bone_names,
            inverse_bind_matrices=result.skin.inverse_bind_matrices,
            config=config,
            _plan=plan,
            _resolved=resolved,
        )

    if result.faces is not None:
        plan.collider_kind = "static"
        plan.source_vertices = len(result.positions)
        return extract_from_mesh(
            result.positions,
            result.faces,
            config=config,
            _plan=plan,
            _resolved=resolved,
        )

    plan.collider_kind = "point_cloud"
    if analysis.opacity_is_logit:
        plan.detected["is_logit"] = True
    return extract_from_arrays(
        result.positions,
        opacity=result.opacity,
        normals=result.normals,
        scales=result.scales,
        rots=result.rots,
        config=config,
        _plan=plan,
        _resolved=resolved,
    )


def extract_from_arrays(
    positions: np.ndarray,
    opacity: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    rots: np.ndarray | None = None,
    config: Config | None = None,
    _plan: BuildPlan | None = None,
    _resolved: ResolvedConfig | None = None,
) -> ExtractionResult:
    config = config or Config()
    positions = np.asarray(positions, dtype=np.float64)
    source_count = len(positions)

    if _plan is None:
        _plan = BuildPlan(input_kind="arrays", collider_kind="point_cloud")
    _plan.source_vertices = source_count

    if _resolved is None:
        analysis = analyze_arrays(positions, opacity, scales, rots)
        _resolved = resolve_config(config, analysis)

    if _resolved.decisions.get("is_environment"):
        _plan.detected["is_environment"] = True

    has_covariance = scales is not None and rots is not None
    if has_covariance:
        scales = np.asarray(scales, dtype=np.float64)
        rots = np.asarray(rots, dtype=np.float64)
        normals = normals_from_covariance(
            scales, rots, log_scale=_resolved.splat_scale_is_log
        )
        _plan.detected["covariance_normals"] = True

    if opacity is not None:
        raw = np.asarray(opacity, dtype=np.float64).ravel()
        if _resolved.opacity_is_logit:
            activated = 1.0 / (1.0 + np.exp(-raw))
        else:
            activated = raw
        mask = activated >= _resolved.opacity_threshold
        positions = positions[mask]
        if normals is not None:
            normals = normals[mask]
        if has_covariance:
            scales = scales[mask]
            rots = rots[mask]
        _plan.step("opacity_filter")
        _plan.detected["filtered_vertices"] = int(mask.sum())

    raw_positions = positions

    if has_covariance and _resolved.splat_surface_ratio > 0:
        if len(positions) > _resolved.spatial_split_threshold:
            _plan.source_vertices = source_count
            return extract_spatial(positions, normals, scales, rots, _resolved, _plan)

        pre_inflate_count = len(positions)
        positions = inflate_splat_points(
            positions,
            scales,
            rots,
            _resolved.splat_surface_ratio,
            log_scale=_resolved.splat_scale_is_log,
        )
        normals = np.tile(normals, (5, 1))
        _plan.step("splat_inflate")
        _plan.detected["inflated_from"] = pre_inflate_count
        _plan.detected["inflated_to"] = len(positions)

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

    result_mesh = poisson_reconstruct(positions, normals, _resolved)
    vertices, triangles = result_mesh

    if len(triangles) < 4:
        return ExtractionResult(
            hulls=[],
            source_vertex_count=source_count,
            mesh_vertex_count=len(vertices),
            build_plan=_plan,
        )

    vertices, triangles = post_poisson_filter(
        vertices, triangles, raw_positions, _resolved
    )
    if len(triangles) < 4:
        return ExtractionResult(
            hulls=[],
            source_vertex_count=source_count,
            mesh_vertex_count=len(vertices),
            build_plan=_plan,
        )

    result = decompose_and_build(
        vertices, triangles, source_count, len(vertices), _resolved, _plan=_plan
    )
    return result


def extract_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: Config | None = None,
    _plan: BuildPlan | None = None,
    _resolved: ResolvedConfig | None = None,
) -> ExtractionResult:
    config = config or Config()
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if _plan is None:
        _plan = BuildPlan(input_kind="mesh", collider_kind="static")
    _plan.source_vertices = _plan.source_vertices or len(vertices)
    if _resolved is None:
        analysis = analyze_arrays(vertices, format="mesh", face_count=len(faces))
        _resolved = resolve_config(config, analysis)
    if len(vertices) < 4 or len(faces) < 4:
        return ExtractionResult(
            hulls=[],
            source_vertex_count=len(vertices),
            mesh_vertex_count=len(vertices),
            build_plan=_plan,
        )
    return decompose_and_build(
        vertices, faces, len(vertices), len(vertices), _resolved, _plan=_plan
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
    _resolved: ResolvedConfig | None = None,
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

    if _resolved is None:
        analysis = analyze_arrays(
            vertices,
            format="mesh",
            face_count=len(faces),
            is_skinned=True,
        )
        _resolved = resolve_config(config, analysis)

    source_count = len(vertices)
    _plan.step("segment_by_bone")
    segments = segment_by_bone(vertices, faces, joint_indices, joint_weights)
    _plan.detected["segment_count"] = len(segments)

    all_hulls = []
    total_mesh_verts = 0
    bones_skipped = 0
    for bone_idx, (seg_verts, seg_faces) in segments.items():
        if len(seg_verts) < _resolved.min_hull_vertices:
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
        result = decompose_and_build(
            seg_verts, seg_faces, len(seg_verts), len(seg_verts), _resolved
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
