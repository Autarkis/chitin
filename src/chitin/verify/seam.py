from __future__ import annotations

import numpy as np

from chitin.phys import PhysHull
from chitin.verify.raycast import ray_closest_hit


def dedup_snags(
    snags: list[tuple[float, float, float]], radius: float
) -> list[tuple[float, float, float]]:
    if not snags:
        return []
    result = [snags[0]]
    for s in snags[1:]:
        if all(
            (s[0] - r[0]) ** 2 + (s[2] - r[2]) ** 2 > radius * radius for r in result
        ):
            result.append(s)
    return result


def _ground_heights(
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    ray_y: float,
    hulls: list[PhysHull],
) -> np.ndarray:
    origins = np.stack([grid_x, np.full_like(grid_x, ray_y), grid_z], axis=1).astype(
        np.float32
    )
    direction = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    closest_t = ray_closest_hit(origins, direction, hulls)
    return np.where(np.isfinite(closest_t), ray_y - closest_t, np.nan)


def find_hull_seam_snags(
    hulls: list,
    grid_resolution: int = 32,
    step_height: float = 0.3,
) -> list[tuple[float, float, float]]:
    if not hulls:
        return []

    phys_hulls = []
    for h in hulls:
        verts = np.asarray(h.vertices, dtype=np.float32)
        phys_hulls.append(
            PhysHull(
                vertices=verts,
                indices=h.indices,
                aabb_min=verts.min(axis=0),
                aabb_max=verts.max(axis=0),
            )
        )

    all_mins = np.array([h.aabb_min for h in phys_hulls])
    all_maxs = np.array([h.aabb_max for h in phys_hulls])
    scene_min = all_mins.min(axis=0)
    scene_max = all_maxs.max(axis=0)
    extent = scene_max - scene_min

    x_range = np.linspace(scene_min[0], scene_max[0], grid_resolution)
    z_range = np.linspace(scene_min[2], scene_max[2], grid_resolution)
    xx, zz = np.meshgrid(x_range, z_range)
    grid_x = xx.ravel()
    grid_z = zz.ravel()

    ray_y = float(scene_max[1] + extent[1] * 0.1)
    heights = _ground_heights(grid_x, grid_z, ray_y, phys_hulls)

    heights_grid = heights.reshape(grid_resolution, grid_resolution)
    ground_grid = np.isfinite(heights_grid)

    snags = []
    for r in range(grid_resolution):
        for c in range(grid_resolution):
            if not ground_grid[r, c]:
                continue
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if nr >= grid_resolution or nc >= grid_resolution:
                    continue
                if not ground_grid[nr, nc]:
                    continue
                h_diff = abs(float(heights_grid[r, c] - heights_grid[nr, nc]))
                if h_diff > step_height:
                    cell_id = r * grid_resolution + c
                    neighbor_id = nr * grid_resolution + nc
                    snag_x = float(grid_x[cell_id] + grid_x[neighbor_id]) / 2
                    snag_y = float(max(heights_grid[r, c], heights_grid[nr, nc]))
                    snag_z = float(grid_z[cell_id] + grid_z[neighbor_id]) / 2
                    snags.append((snag_x, snag_y, snag_z))

    return dedup_snags(snags, step_height)
