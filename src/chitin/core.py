# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


import coacd
import numpy as np
import trimesh

from chitin.analyze import analyze_arrays
from chitin.config import Config
from chitin.plan import BuildPlan
from chitin.resolve import ResolvedConfig, resolve_config
from chitin.result import BoneInfo, ExtractionResult, Hull, LodHulls


def _quat_to_rotation_matrices(rots: np.ndarray) -> np.ndarray:
    n = len(rots)
    w, x, y, z = rots[:, 0], rots[:, 1], rots[:, 2], rots[:, 3]
    norms = np.sqrt(w * w + x * x + y * y + z * z)
    norms = np.where(norms == 0, 1.0, norms)
    w, x, y, z = w / norms, x / norms, y / norms, z / norms

    R = np.zeros((n, 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _normals_from_covariance(
    scales: np.ndarray, rots: np.ndarray, log_scale: bool = True
) -> np.ndarray:
    linear_scales = np.exp(scales) if log_scale else scales
    R = _quat_to_rotation_matrices(rots)
    min_axis = np.argmin(linear_scales, axis=1)
    normals = R[np.arange(len(scales)), :, min_axis]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return normals / norms


def _inflate_splat_points(
    positions: np.ndarray,
    scales: np.ndarray,
    rots: np.ndarray,
    surface_ratio: float,
    log_scale: bool = True,
) -> np.ndarray:
    """Expand gaussian centers into disk samples along the two largest axes."""
    linear_scales = np.exp(scales) if log_scale else scales
    R = _quat_to_rotation_matrices(rots)

    sorted_axes = np.argsort(linear_scales, axis=1)
    major = sorted_axes[:, 2]
    minor = sorted_axes[:, 1]

    n = len(positions)
    idx = np.arange(n)
    axis_a = R[idx, :, major] * (linear_scales[idx, major, np.newaxis] * surface_ratio)
    axis_b = R[idx, :, minor] * (linear_scales[idx, minor, np.newaxis] * surface_ratio)

    samples = [positions]
    samples.append(positions + axis_a)
    samples.append(positions - axis_a)
    samples.append(positions + axis_b)
    samples.append(positions - axis_b)
    return np.concatenate(samples, axis=0)


@dataclass
class _OctreeCell:
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    indices: np.ndarray


def _octree_partition(
    positions: np.ndarray, max_points: int, max_depth: int = 6
) -> list[_OctreeCell]:
    cells: list[_OctreeCell] = []

    def _split(indices: np.ndarray, bmin: np.ndarray, bmax: np.ndarray, depth: int):
        if len(indices) <= max_points or depth >= max_depth:
            cells.append(_OctreeCell(bounds_min=bmin, bounds_max=bmax, indices=indices))
            return
        mid = (bmin + bmax) / 2
        pts = positions[indices]
        octant_id = (
            (pts[:, 0] >= mid[0]).astype(np.int32)
            | ((pts[:, 1] >= mid[1]).astype(np.int32) << 1)
            | ((pts[:, 2] >= mid[2]).astype(np.int32) << 2)
        )
        for octant in range(8):
            child_mask = octant_id == octant
            child_indices = indices[child_mask]
            if len(child_indices) == 0:
                continue
            child_min = np.array(
                [
                    mid[0] if (octant & 1) else bmin[0],
                    mid[1] if (octant & 2) else bmin[1],
                    mid[2] if (octant & 4) else bmin[2],
                ]
            )
            child_max = np.array(
                [
                    bmax[0] if (octant & 1) else mid[0],
                    bmax[1] if (octant & 2) else mid[1],
                    bmax[2] if (octant & 4) else mid[2],
                ]
            )
            _split(child_indices, child_min, child_max, depth + 1)

    scene_min = positions.min(axis=0)
    scene_max = positions.max(axis=0)
    extent = scene_max - scene_min
    extent = np.where(extent == 0, 1.0, extent)
    scene_max = scene_min + extent

    _split(np.arange(len(positions)), scene_min, scene_max, 0)
    return cells


def _aabb_overlaps_bounds(
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


def _aabb_iou(a: Hull, b: Hull) -> float:
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


def _dedup_overlapping_hulls(
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
            if _aabb_iou(hulls[i], hulls[j]) >= iou_threshold:
                discarded.add(j)
    return kept


def _uf_find(parent: dict[int, int], x: int) -> int:
    if x not in parent:
        parent[x] = x
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _uf_union(parent: dict[int, int], a: int, b: int) -> None:
    ra, rb = _uf_find(parent, a), _uf_find(parent, b)
    if ra != rb:
        parent[rb] = ra


def _extract_cell_hulls(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    positions: np.ndarray,
    normals: np.ndarray,
    scales: np.ndarray | None,
    rots: np.ndarray | None,
    max_radii: np.ndarray,
    config: ResolvedConfig,
) -> list[Hull]:
    strict_mask = np.all((positions >= bounds_min) & (positions <= bounds_max), axis=1)
    strict_radii = max_radii[strict_mask]
    if len(strict_radii) == 0:
        return []
    padding = float(np.percentile(strict_radii, 95)) * 3.0

    padded_min = bounds_min - padding
    padded_max = bounds_max + padding
    padded_mask = np.all((positions >= padded_min) & (positions <= padded_max), axis=1)
    cell_positions = positions[padded_mask]
    cell_normals = normals[padded_mask]

    if len(cell_positions) < 100:
        return []

    raw_cell_positions = cell_positions
    if scales is not None and rots is not None and config.splat_surface_ratio > 0:
        cell_scales = scales[padded_mask]
        cell_rots = rots[padded_mask]
        cell_positions = _inflate_splat_points(
            cell_positions,
            cell_scales,
            cell_rots,
            config.splat_surface_ratio,
            log_scale=config.splat_scale_is_log,
        )
        cell_normals = np.tile(cell_normals, (5, 1))

    result_mesh = _poisson_reconstruct(
        cell_positions, cell_normals, config, isolate=True
    )
    if result_mesh is None:
        return []
    verts, tris = result_mesh
    if len(tris) < 4:
        return []

    verts, tris = _post_poisson_filter(verts, tris, raw_cell_positions, config)
    if len(tris) < 4:
        return []

    is_flat, flat_normal = (
        _is_flat_mesh(verts, tris, config.flatness_threshold)
        if config.flatness_threshold > 0
        else (False, None)
    )
    if is_flat:
        hull = _make_planar_box(verts, flat_normal)
        if _aabb_overlaps_bounds(hull, bounds_min, bounds_max):
            return [hull]
        return []

    result = _decompose_and_build(verts, tris, len(cell_positions), len(verts), config)
    return [h for h in result.hulls if _aabb_overlaps_bounds(h, bounds_min, bounds_max)]


def _seam_repair_pass(
    all_hulls: list[Hull],
    cells: list[_OctreeCell],
    hull_cell_map: list[int],
    positions: np.ndarray,
    normals: np.ndarray,
    scales: np.ndarray | None,
    rots: np.ndarray | None,
    max_radii: np.ndarray,
    config: ResolvedConfig,
) -> list[Hull]:
    from chitin.sweep import find_hull_seam_snags

    snags = find_hull_seam_snags(all_hulls)
    if not snags:
        return all_hulls

    cell_sizes = np.array([c.bounds_max - c.bounds_min for c in cells])
    margin = float(np.median(cell_sizes)) * 0.1

    parent: dict[int, int] = {}
    for sx, _sy, sz in snags:
        nearby = []
        for ci, cell in enumerate(cells):
            if (
                cell.bounds_min[0] - margin <= sx <= cell.bounds_max[0] + margin
                and cell.bounds_min[2] - margin <= sz <= cell.bounds_max[2] + margin
            ):
                nearby.append(ci)
        if len(nearby) >= 2:
            root = _uf_find(parent, nearby[0])
            for ci in nearby[1:]:
                _uf_union(parent, root, ci)

    groups: dict[int, list[int]] = {}
    for ci in parent:
        root = _uf_find(parent, ci)
        groups.setdefault(root, []).append(ci)
    merge_groups = [g for g in groups.values() if len(g) >= 2]

    if not merge_groups:
        return all_hulls

    cells_to_repair = set()
    for g in merge_groups:
        cells_to_repair.update(g)

    remove_set = {i for i, ci in enumerate(hull_cell_map) if ci in cells_to_repair}
    new_hulls = [h for i, h in enumerate(all_hulls) if i not in remove_set]

    for group in merge_groups:
        merged_min = np.min([cells[ci].bounds_min for ci in group], axis=0)
        merged_max = np.max([cells[ci].bounds_max for ci in group], axis=0)
        repaired = _extract_cell_hulls(
            merged_min,
            merged_max,
            positions,
            normals,
            scales,
            rots,
            max_radii,
            config,
        )
        new_hulls.extend(repaired)

    return new_hulls


def _extract_spatial(
    positions: np.ndarray,
    normals: np.ndarray,
    scales: np.ndarray,
    rots: np.ndarray,
    config: ResolvedConfig,
    plan: BuildPlan,
) -> ExtractionResult:
    linear_scales = np.exp(scales) if config.splat_scale_is_log else scales
    max_radii = np.max(linear_scales, axis=1)

    cells = _octree_partition(positions, config.spatial_split_threshold)
    plan.step("spatial_partition")
    plan.detected["cell_count"] = len(cells)

    source_count = plan.source_vertices or len(positions)
    all_hulls: list[Hull] = []
    hull_cell_map: list[int] = []
    lod_buckets: dict[int, list[Hull]] = {}
    cell_paddings: list[float] = []

    for cell_idx, cell in enumerate(cells):
        cell_radii = max_radii[cell.indices]
        cell_p95 = float(np.percentile(cell_radii, 95)) if len(cell_radii) > 0 else 0
        padding = cell_p95 * 3.0
        cell_paddings.append(padding)
        padded_min = cell.bounds_min - padding
        padded_max = cell.bounds_max + padding
        padded_mask = (
            (positions[:, 0] >= padded_min[0])
            & (positions[:, 0] <= padded_max[0])
            & (positions[:, 1] >= padded_min[1])
            & (positions[:, 1] <= padded_max[1])
            & (positions[:, 2] >= padded_min[2])
            & (positions[:, 2] <= padded_max[2])
        )
        cell_positions = positions[padded_mask]
        cell_normals = normals[padded_mask]
        cell_scales = scales[padded_mask]
        cell_rots = rots[padded_mask]

        if len(cell_positions) < 100:
            continue

        raw_cell_positions = cell_positions
        if config.splat_surface_ratio > 0:
            cell_positions = _inflate_splat_points(
                cell_positions,
                cell_scales,
                cell_rots,
                config.splat_surface_ratio,
                log_scale=config.splat_scale_is_log,
            )
            cell_normals = np.tile(cell_normals, (5, 1))

        cell_result_mesh = _poisson_reconstruct(
            cell_positions, cell_normals, config, isolate=True
        )
        if cell_result_mesh is None:
            continue
        cell_verts, cell_tris = cell_result_mesh

        if len(cell_tris) < 4:
            continue

        cell_verts, cell_tris = _post_poisson_filter(
            cell_verts, cell_tris, raw_cell_positions, config
        )
        if len(cell_tris) < 4:
            continue

        is_flat, flat_normal = (
            _is_flat_mesh(cell_verts, cell_tris, config.flatness_threshold)
            if config.flatness_threshold > 0
            else (False, None)
        )
        if is_flat:
            box_hull = _make_planar_box(cell_verts, flat_normal)
            cell_result = ExtractionResult(
                hulls=[box_hull],
                source_vertex_count=len(cell_positions),
                mesh_vertex_count=len(cell_verts),
            )
        else:
            cell_result = _decompose_and_build(
                cell_verts,
                cell_tris,
                len(cell_positions),
                len(cell_verts),
                config,
            )

        strict_min = cell.bounds_min
        strict_max = cell.bounds_max

        for hull in cell_result.hulls:
            if _aabb_overlaps_bounds(hull, strict_min, strict_max):
                all_hulls.append(hull)
                hull_cell_map.append(cell_idx)

        if cell_result.lod_tiers:
            for tier_idx, tier in enumerate(cell_result.lod_tiers):
                if tier_idx not in lod_buckets:
                    lod_buckets[tier_idx] = []
                for hull in tier.hulls:
                    if _aabb_overlaps_bounds(hull, strict_min, strict_max):
                        lod_buckets[tier_idx].append(hull)

    if config.seam_repair and len(cells) > 1:
        pre_repair = len(all_hulls)
        all_hulls = _seam_repair_pass(
            all_hulls,
            cells,
            hull_cell_map,
            positions,
            normals,
            scales,
            rots,
            max_radii,
            config,
        )
        repaired_count = pre_repair - len(all_hulls)
        if repaired_count != 0:
            plan.detected["seam_repair_delta"] = len(all_hulls) - pre_repair

    all_hulls = _dedup_overlapping_hulls(all_hulls)
    for tier_idx in lod_buckets:
        lod_buckets[tier_idx] = _dedup_overlapping_hulls(lod_buckets[tier_idx])

    plan.step("spatial_reconcile")
    plan.detected["reconciled_hulls"] = len(all_hulls)
    if cell_paddings:
        plan.detected["padding_min"] = float(np.min(cell_paddings))
        plan.detected["padding_median"] = float(np.median(cell_paddings))
        plan.detected["padding_max"] = float(np.max(cell_paddings))

    merged_lod_tiers = None
    if lod_buckets and config.lod_concavities:
        merged_lod_tiers = []
        sorted_concavities = sorted(config.lod_concavities)
        for tier_idx in sorted(lod_buckets.keys()):
            concavity = (
                sorted_concavities[tier_idx]
                if tier_idx < len(sorted_concavities)
                else 0.0
            )
            merged_lod_tiers.append(
                LodHulls(concavity=concavity, hulls=lod_buckets[tier_idx])
            )

    return ExtractionResult(
        hulls=all_hulls,
        source_vertex_count=source_count,
        mesh_vertex_count=sum(len(h.vertices) for h in all_hulls),
        build_plan=plan,
        lod_tiers=merged_lod_tiers,
    )


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
        fmt = suffix.lstrip(".")
        analysis = analyze_arrays(verts, format=fmt, face_count=len(faces))
        resolved = resolve_config(config, analysis)
        del mesh
        return extract_from_mesh(
            verts, faces, config=config, _plan=plan, _resolved=resolved
        )
    if suffix in (".glb", ".gltf", ".fbx"):
        return _extract_from_skinned_or_static(path, config)
    if suffix in (".usd", ".usda", ".usdc"):
        return _extract_from_usd(path, config)

    raise ValueError(f"Unsupported input format: {suffix}")


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
        normals = _normals_from_covariance(
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
            return _extract_spatial(positions, normals, scales, rots, _resolved, _plan)

        pre_inflate_count = len(positions)
        positions = _inflate_splat_points(
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

    result_mesh = _poisson_reconstruct(positions, normals, _resolved)
    vertices, triangles = result_mesh

    if len(triangles) < 4:
        return ExtractionResult(
            hulls=[],
            source_vertex_count=source_count,
            mesh_vertex_count=len(vertices),
            build_plan=_plan,
        )

    vertices, triangles = _post_poisson_filter(
        vertices, triangles, raw_positions, _resolved
    )
    if len(triangles) < 4:
        return ExtractionResult(
            hulls=[],
            source_vertex_count=source_count,
            mesh_vertex_count=len(vertices),
            build_plan=_plan,
        )

    result = _decompose_and_build(
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
    return _decompose_and_build(
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
    segments = _segment_by_bone(vertices, faces, joint_indices, joint_weights)
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
        result = _decompose_and_build(
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
            analysis = analyze_arrays(
                verts, format=suffix, face_count=len(faces), is_skinned=True
            )
            resolved = resolve_config(config, analysis)
            return extract_from_rigged_mesh(
                verts,
                faces,
                np.asarray(skin_data.joint_indices, dtype=np.int32),
                np.asarray(skin_data.joint_weights, dtype=np.float64),
                bone_names=skin_data.joint_names,
                inverse_bind_matrices=skin_data.inverse_bind_matrices,
                config=config,
                _plan=plan,
                _resolved=resolved,
            )

    plan.detected["is_skinned"] = False
    plan.collider_kind = "static"

    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry()
        if isinstance(mesh, trimesh.Trimesh):
            mesh.visual = trimesh.visual.ColorVisuals()
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            analysis = analyze_arrays(verts, format=suffix, face_count=len(faces))
            resolved = resolve_config(config, analysis)
            del mesh
            return extract_from_mesh(
                verts, faces, config=config, _plan=plan, _resolved=resolved
            )
        return ExtractionResult(
            hulls=[], source_vertex_count=0, mesh_vertex_count=0, build_plan=plan
        )

    loaded.visual = trimesh.visual.ColorVisuals()
    verts = np.asarray(loaded.vertices, dtype=np.float32)
    faces = np.asarray(loaded.faces, dtype=np.int32)
    analysis = analyze_arrays(verts, format=suffix, face_count=len(faces))
    resolved = resolve_config(config, analysis)
    del loaded
    return extract_from_mesh(
        verts, faces, config=config, _plan=plan, _resolved=resolved
    )


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
    has_opacity = "opacity" in vertex.data.dtype.names
    if has_opacity:
        opacity = np.asarray(vertex["opacity"], dtype=np.float64)

    has_scales = all(f"scale_{i}" in vertex.data.dtype.names for i in range(3))
    has_rots = all(f"rot_{i}" in vertex.data.dtype.names for i in range(4))
    has_covariance = has_scales and has_rots

    normals = None
    scales_arr = None
    rots_arr = None
    if has_covariance:
        scales_arr = np.column_stack([vertex[f"scale_{i}"] for i in range(3)]).astype(
            np.float64
        )
        rots_arr = np.column_stack([vertex[f"rot_{i}"] for i in range(4)]).astype(
            np.float64
        )
    else:
        has_normals = all(n in vertex.data.dtype.names for n in ("nx", "ny", "nz"))
        if has_normals:
            normals = np.column_stack(
                [vertex["nx"], vertex["ny"], vertex["nz"]]
            ).astype(np.float64)

    analysis = analyze_arrays(positions, opacity, scales_arr, rots_arr, format="ply")
    resolved = resolve_config(config, analysis)

    plan.detected["has_opacity"] = has_opacity
    plan.detected["is_logit"] = analysis.opacity_is_logit
    plan.detected["has_normals"] = normals is not None or has_covariance
    plan.detected["has_covariance"] = has_covariance

    return extract_from_arrays(
        positions,
        opacity=opacity,
        normals=normals,
        scales=scales_arr,
        rots=rots_arr,
        config=config,
        _plan=plan,
        _resolved=resolved,
    )


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
    analysis = analyze_arrays(vertices, format=suffix, face_count=len(faces))
    resolved = resolve_config(config, analysis)
    return extract_from_mesh(vertices, faces, config, _plan=plan, _resolved=resolved)


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


_POISSON_WORKER_SCRIPT = Path(__file__).parent / "_poisson_worker.py"


def _poisson_reconstruct_inner(
    positions: np.ndarray,
    normals: np.ndarray | None,
    depth: int,
    density_quantile: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "Point cloud extraction requires open3d. "
            "Install with: pip install chitin[splat]"
        )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)

    if normals is not None and not np.allclose(normals, 0):
        pcd.normals = o3d.utility.Vector3dVector(normals)
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=10)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    densities = np.asarray(densities)
    if len(densities) > 0:
        density_threshold = np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(densities < density_threshold)

    return (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.triangles, dtype=np.int32),
    )


def _poisson_reconstruct(
    positions: np.ndarray,
    normals: np.ndarray | None,
    config: ResolvedConfig,
    isolate: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    depth = config.poisson_depth
    dq = config.poisson_density_quantile

    if not isolate:
        return _poisson_reconstruct_inner(positions, normals, depth, dq)

    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "input.npz"
        out_path = Path(tmpdir) / "output.npz"

        save_dict: dict[str, np.ndarray] = {
            "positions": positions,
            "depth": np.array([depth]),
            "density_quantile": np.array([dq]),
        }
        if normals is not None:
            save_dict["normals"] = normals
        np.savez(in_path, **save_dict)

        result = subprocess.run(
            [sys.executable, str(_POISSON_WORKER_SCRIPT), str(in_path), str(out_path)],
            capture_output=True,
            timeout=300,
        )

        if result.returncode != 0 or not out_path.exists():
            return None

        data = np.load(out_path)
        return data["vertices"], data["triangles"]


def _proximity_filter_mesh(
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


def _is_flat_mesh(
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


def _make_planar_box(vertices: np.ndarray, dominant_normal: np.ndarray) -> Hull:
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


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    vnormals = np.zeros_like(vertices)
    np.add.at(vnormals, faces[:, 0], face_normals)
    np.add.at(vnormals, faces[:, 1], face_normals)
    np.add.at(vnormals, faces[:, 2], face_normals)
    norms = np.linalg.norm(vnormals, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vnormals / norms


def _extrude_thin_shell(
    vertices: np.ndarray,
    faces: np.ndarray,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray]:
    vnormals = _vertex_normals(vertices, faces)

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


def _post_poisson_filter(
    mesh_verts: np.ndarray,
    mesh_tris: np.ndarray,
    input_positions: np.ndarray,
    config: ResolvedConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if config.surface_proximity_filter > 0:
        mesh_verts, mesh_tris = _proximity_filter_mesh(
            mesh_verts, mesh_tris, input_positions, config.surface_proximity_filter
        )
    if config.thin_shell and len(mesh_tris) >= 4:
        thickness = config.thin_shell_thickness
        if thickness <= 0:
            extent = mesh_verts.max(axis=0) - mesh_verts.min(axis=0)
            thickness = np.median(extent) * 0.02
        mesh_verts, mesh_tris = _extrude_thin_shell(mesh_verts, mesh_tris, thickness)
    return mesh_verts, mesh_tris


def _maybe_decimate(
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


def _extract_walkable_hulls(
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

        box_hull = _make_planar_box(comp_verts, avg_normal)
        hulls.append(box_hull)

    final_unwalkable = (
        np.vstack(keep_unwalkable)
        if keep_unwalkable
        else np.array([], dtype=np.int32).reshape(0, 3)
    )
    return hulls, final_unwalkable


def _decompose_and_build(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_count: int,
    mesh_count: int,
    config: ResolvedConfig,
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

    is_env = _plan is not None and _plan.detected.get("is_environment", False)
    env_hulls = []
    if is_env:
        env_hulls, faces = _extract_walkable_hulls(vertices, faces)
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
