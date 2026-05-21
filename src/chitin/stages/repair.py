# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

import numpy as np

from chitin.resolve import ResolvedConfig
from chitin.result import Hull
from chitin.stages.decompose import aabb_overlaps_bounds
from chitin.stages.filter import post_poisson_filter
from chitin.stages.flatness import is_flat_mesh, make_planar_box
from chitin.stages.reconstruct import poisson_reconstruct
from chitin.stages.splat import OctreeCell, inflate_splat_points


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


def extract_cell_hulls(
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
        cell_positions = inflate_splat_points(
            cell_positions,
            cell_scales,
            cell_rots,
            config.splat_surface_ratio,
            log_scale=config.splat_scale_is_log,
        )
        cell_normals = np.tile(cell_normals, (5, 1))

    result_mesh = poisson_reconstruct(
        cell_positions, cell_normals, config, isolate=True
    )
    if result_mesh is None:
        return []
    verts, tris = result_mesh
    if len(tris) < 4:
        return []

    verts, tris = post_poisson_filter(verts, tris, raw_cell_positions, config)
    if len(tris) < 4:
        return []

    flat, flat_normal = (
        is_flat_mesh(verts, tris, config.flatness_threshold)
        if config.flatness_threshold > 0
        else (False, None)
    )
    if flat:
        hull = make_planar_box(verts, flat_normal)
        if aabb_overlaps_bounds(hull, bounds_min, bounds_max):
            return [hull]
        return []

    from chitin.stages.decompose import decompose_and_build

    result = decompose_and_build(verts, tris, len(cell_positions), len(verts), config)
    return [h for h in result.hulls if aabb_overlaps_bounds(h, bounds_min, bounds_max)]


def seam_repair_pass(
    all_hulls: list[Hull],
    cells: list[OctreeCell],
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
        for i in range(len(nearby)):
            for j in range(i + 1, len(nearby)):
                _uf_union(parent, nearby[i], nearby[j])

    groups: dict[int, list[int]] = {}
    for ci in range(len(cells)):
        root = _uf_find(parent, ci)
        groups.setdefault(root, []).append(ci)

    merged_groups = [g for g in groups.values() if len(g) > 1]
    if not merged_groups:
        return all_hulls

    merged_cell_indices: set[int] = set()
    for g in merged_groups:
        merged_cell_indices.update(g)

    new_hulls = [
        h
        for i, h in enumerate(all_hulls)
        if i >= len(hull_cell_map) or hull_cell_map[i] not in merged_cell_indices
    ]

    for group in merged_groups:
        merged_min = np.min([cells[ci].bounds_min for ci in group], axis=0)
        merged_max = np.max([cells[ci].bounds_max for ci in group], axis=0)
        repaired = extract_cell_hulls(
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
