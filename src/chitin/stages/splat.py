from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chitin.plan import BuildPlan
from chitin.resolve import ResolvedConfig
from chitin.result import ExtractionResult, Hull, LodHulls
from chitin.stages.decompose import (
    aabb_overlaps_bounds,
    decompose_and_build,
    dedup_overlapping_hulls,
)
from chitin.stages.filter import post_poisson_filter
from chitin.stages.flatness import is_flat_mesh, make_planar_box
from chitin.stages.reconstruct import PoissonWorkerError, poisson_reconstruct


def quat_to_rotation_matrices(rots: np.ndarray) -> np.ndarray:
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


def normals_from_covariance(
    scales: np.ndarray, rots: np.ndarray, log_scale: bool = True
) -> np.ndarray:
    linear_scales = np.exp(scales) if log_scale else scales
    R = quat_to_rotation_matrices(rots)
    min_axis = np.argmin(linear_scales, axis=1)
    normals = R[np.arange(len(scales)), :, min_axis]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return normals / norms


def orient_normals_consistently(
    positions: np.ndarray, normals: np.ndarray, k: int = 10
) -> np.ndarray:
    """Sign-correct normals via consistent tangent-plane propagation.

    Covariance normals carry the splat's minor-axis direction but an
    arbitrary sign, and Poisson reconstruction is sensitive to flips.
    Keeps the directions, flips signs only (Hoppe et al. 1992 MST
    propagation, via Open3D). No-op when open3d is unavailable.
    """
    try:
        import open3d as o3d
    except ImportError:
        return normals
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.orient_normals_consistent_tangent_plane(k)
    oriented = np.asarray(pcd.normals)
    flip = np.einsum("ij,ij->i", oriented, normals) < 0
    out = normals.copy()
    out[flip] = -out[flip]
    return out


def inflate_splat_points(
    positions: np.ndarray,
    scales: np.ndarray,
    rots: np.ndarray,
    surface_ratio: float,
    log_scale: bool = True,
) -> np.ndarray:
    linear_scales = np.exp(scales) if log_scale else scales
    R = quat_to_rotation_matrices(rots)

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
class OctreeCell:
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    indices: np.ndarray


def octree_partition(
    positions: np.ndarray, max_points: int, max_depth: int = 6
) -> list[OctreeCell]:
    cells: list[OctreeCell] = []

    def _split(indices: np.ndarray, bmin: np.ndarray, bmax: np.ndarray, depth: int):
        if len(indices) <= max_points or depth >= max_depth:
            cells.append(OctreeCell(bounds_min=bmin, bounds_max=bmax, indices=indices))
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


def _process_single_cell(
    cell_positions: np.ndarray,
    cell_normals: np.ndarray,
    cell_scales: np.ndarray,
    cell_rots: np.ndarray,
    strict_min: np.ndarray,
    strict_max: np.ndarray,
    config: ResolvedConfig,
) -> tuple[list[Hull], list[tuple[int, list[Hull]]]] | str:
    """Process one padded octree cell. Returns (hulls, lod_entries) on
    success, or a failure-reason string for the build plan."""
    raw_cell_positions = cell_positions
    if config.splat_surface_ratio > 0:
        cell_positions = inflate_splat_points(
            cell_positions,
            cell_scales,
            cell_rots,
            config.splat_surface_ratio,
            log_scale=config.splat_scale_is_log,
        )
        cell_normals = np.tile(cell_normals, (5, 1))

    try:
        cell_result_mesh = poisson_reconstruct(
            cell_positions, cell_normals, config, isolate=True
        )
    except PoissonWorkerError as exc:
        return f"poisson_failed: {exc}"
    cell_verts, cell_tris = cell_result_mesh

    if len(cell_tris) < 4:
        return "too_few_triangles"

    cell_verts, cell_tris = post_poisson_filter(
        cell_verts, cell_tris, raw_cell_positions, config
    )
    if len(cell_tris) < 4:
        return "filtered_out"

    flat, flat_normal = (
        is_flat_mesh(cell_verts, cell_tris, config.flatness_threshold)
        if config.flatness_threshold > 0
        else (False, None)
    )
    if flat:
        box_hull = make_planar_box(cell_verts, flat_normal)
        cell_result = ExtractionResult(
            hulls=[box_hull],
            source_vertex_count=len(cell_positions),
            mesh_vertex_count=len(cell_verts),
        )
    else:
        cell_result = decompose_and_build(
            cell_verts,
            cell_tris,
            len(cell_positions),
            len(cell_verts),
            config,
        )

    hulls = [
        h for h in cell_result.hulls if aabb_overlaps_bounds(h, strict_min, strict_max)
    ]
    lod_entries: list[tuple[int, list[Hull]]] = []
    if cell_result.lod_tiers:
        for tier_idx, tier in enumerate(cell_result.lod_tiers):
            tier_hulls = [
                h for h in tier.hulls if aabb_overlaps_bounds(h, strict_min, strict_max)
            ]
            if tier_hulls:
                lod_entries.append((tier_idx, tier_hulls))

    return hulls, lod_entries


def extract_spatial(
    positions: np.ndarray,
    normals: np.ndarray,
    scales: np.ndarray,
    rots: np.ndarray,
    config: ResolvedConfig,
    plan: BuildPlan,
) -> ExtractionResult:
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    linear_scales = np.exp(scales) if config.splat_scale_is_log else scales
    max_radii = np.max(linear_scales, axis=1)

    cells = octree_partition(positions, config.spatial_split_threshold)
    plan.step("spatial_partition")
    plan.detected["cell_count"] = len(cells)

    source_count = plan.source_vertices or len(positions)
    all_hulls: list[Hull] = []
    hull_cell_map: list[int] = []
    lod_buckets: dict[int, list[Hull]] = {}
    cell_paddings: list[float] = []

    cell_tasks: list[
        tuple[
            int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ]
    ] = []
    cells_skipped_sparse = 0
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
        if len(cell_positions) < 100:
            cells_skipped_sparse += 1
            continue
        cell_normals = normals[padded_mask]
        cell_scales = scales[padded_mask]
        cell_rots = rots[padded_mask]
        cell_tasks.append(
            (
                cell_idx,
                cell_positions,
                cell_normals,
                cell_scales,
                cell_rots,
                cell.bounds_min.copy(),
                cell.bounds_max.copy(),
            )
        )

    max_workers = min(os.cpu_count() or 1, len(cell_tasks), 8)
    plan.detected["parallel_workers"] = max_workers
    plan.detected["cells_skipped_sparse"] = cells_skipped_sparse
    cells_failed = 0
    failure_reasons: dict[str, int] = {}
    failed_cells: list[tuple[int, str]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for cell_idx, c_pos, c_norm, c_sc, c_rot, s_min, s_max in cell_tasks:
            f = executor.submit(
                _process_single_cell,
                c_pos,
                c_norm,
                c_sc,
                c_rot,
                s_min,
                s_max,
                config,
            )
            futures[f] = cell_idx

        for future in as_completed(futures):
            cell_idx = futures[future]
            result = future.result()
            if isinstance(result, str):
                failed_cells.append((cell_idx, result))
                continue
            hulls, lod_entries = result
            for hull in hulls:
                all_hulls.append(hull)
                hull_cell_map.append(cell_idx)
            for tier_idx, tier_hulls in lod_entries:
                if tier_idx not in lod_buckets:
                    lod_buckets[tier_idx] = []
                lod_buckets[tier_idx].extend(tier_hulls)

    if config.seam_repair and len(cells) > 1:
        from chitin.stages.repair import seam_repair_pass

        pre_repair = len(all_hulls)
        all_hulls = seam_repair_pass(
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

    # Serial retry for poisson failures: worker segfaults are sporadic and
    # load-dependent, so a second attempt without pool contention usually
    # succeeds. Other failure kinds are deterministic; don't retry those.
    task_by_idx = {t[0]: t for t in cell_tasks}
    cells_retried = 0
    for cell_idx, reason in failed_cells:
        retried = None
        if reason.startswith("poisson_failed") and cell_idx in task_by_idx:
            cells_retried += 1
            _, c_pos, c_norm, c_sc, c_rot, s_min, s_max = task_by_idx[cell_idx]
            retried = _process_single_cell(
                c_pos, c_norm, c_sc, c_rot, s_min, s_max, config
            )
        if retried is None or isinstance(retried, str):
            final_reason = retried if isinstance(retried, str) else reason
            cells_failed += 1
            failure_reasons[final_reason] = failure_reasons.get(final_reason, 0) + 1
            continue
        retry_hulls, retry_lod_entries = retried
        for hull in retry_hulls:
            all_hulls.append(hull)
            hull_cell_map.append(cell_idx)
        for tier_idx, tier_hulls in retry_lod_entries:
            if tier_idx not in lod_buckets:
                lod_buckets[tier_idx] = []
            lod_buckets[tier_idx].extend(tier_hulls)
    if cells_retried:
        plan.detected["cells_retried"] = cells_retried

    plan.detected["cells_failed"] = cells_failed
    if failure_reasons:
        plan.detected["cell_failure_reasons"] = failure_reasons

    pre_dedup = len(all_hulls)
    all_hulls = dedup_overlapping_hulls(all_hulls)
    plan.detected["dedup_removed"] = pre_dedup - len(all_hulls)
    for tier_idx in lod_buckets:
        lod_buckets[tier_idx] = dedup_overlapping_hulls(lod_buckets[tier_idx])

    from chitin.stages.decompose import (
        consolidate_near_contained_hulls,
        cull_contained_hulls,
    )

    pre_cull = len(all_hulls)
    all_hulls = cull_contained_hulls(all_hulls)
    plan.detected["containment_culled"] = pre_cull - len(all_hulls)
    pre_consolidate = len(all_hulls)
    all_hulls = consolidate_near_contained_hulls(all_hulls)
    plan.detected["consolidated"] = pre_consolidate - len(all_hulls)

    from chitin.stages.occlusion import cull_occluded_hulls

    all_hulls, occlusion_culled = cull_occluded_hulls(all_hulls, positions)
    plan.detected["occlusion_culled"] = occlusion_culled
    for tier_idx in lod_buckets:
        lod_buckets[tier_idx], _ = cull_occluded_hulls(lod_buckets[tier_idx], positions)
    for tier_idx in lod_buckets:
        lod_buckets[tier_idx] = cull_contained_hulls(lod_buckets[tier_idx])
        lod_buckets[tier_idx] = consolidate_near_contained_hulls(lod_buckets[tier_idx])

    plan.step("spatial_reconcile")
    plan.detected["reconciled_hulls"] = len(all_hulls)

    from chitin.verify.coverage import coverage_report

    plan.step("coverage")
    plan.detected["coverage"] = coverage_report(
        all_hulls, positions, cell_indices=[cell.indices for cell in cells]
    )
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
