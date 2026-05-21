# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chitin.phys import read_phys
from chitin.verify.raycast import ray_closest_hit
from chitin.verify.seam import dedup_snags


@dataclass
class SweepResult:
    grid_resolution: int
    total_cells: int
    ground_cells: int
    connected_components: int
    largest_component: int
    traversability: float
    island_sizes: list[int]
    seam_snags: list[tuple[float, float, float]]
    capsule_radius: float
    capsule_height: float
    step_height: float
    scene_aabb_min: np.ndarray
    scene_aabb_max: np.ndarray

    @property
    def rating(self) -> str:
        if self.traversability >= 0.95:
            return "excellent"
        if self.traversability >= 0.80:
            return "good"
        if self.traversability >= 0.50:
            return "fair"
        return "poor"

    def to_json(self, path: str | Path) -> None:
        data = {
            "grid_resolution": self.grid_resolution,
            "total_cells": self.total_cells,
            "ground_cells": self.ground_cells,
            "connected_components": self.connected_components,
            "largest_component": self.largest_component,
            "traversability": round(self.traversability, 4),
            "rating": self.rating,
            "island_sizes": self.island_sizes,
            "seam_snags": self.seam_snags,
            "capsule_radius": self.capsule_radius,
            "capsule_height": self.capsule_height,
            "step_height": self.step_height,
            "scene_aabb_min": self.scene_aabb_min.tolist(),
            "scene_aabb_max": self.scene_aabb_max.tolist(),
        }
        Path(path).write_text(json.dumps(data, indent=2))


def _ground_heights(
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    ray_y: float,
    hulls: list,
) -> np.ndarray:
    origins = np.stack([grid_x, np.full_like(grid_x, ray_y), grid_z], axis=1).astype(
        np.float32
    )
    direction = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    closest_t = ray_closest_hit(origins, direction, hulls)
    return np.where(np.isfinite(closest_t), ray_y - closest_t, np.nan)


def sweep(
    phys_path: str | Path,
    grid_resolution: int = 32,
    capsule_radius: float = 0.3,
    capsule_height: float = 1.8,
    step_height: float = 0.3,
) -> SweepResult:
    pf = read_phys(phys_path)

    if not pf.hulls:
        return SweepResult(
            grid_resolution=grid_resolution,
            total_cells=0,
            ground_cells=0,
            connected_components=0,
            largest_component=0,
            traversability=0.0,
            island_sizes=[],
            seam_snags=[],
            capsule_radius=capsule_radius,
            capsule_height=capsule_height,
            step_height=step_height,
            scene_aabb_min=np.zeros(3),
            scene_aabb_max=np.zeros(3),
        )

    all_mins = np.array([h.aabb_min for h in pf.hulls])
    all_maxs = np.array([h.aabb_max for h in pf.hulls])
    scene_min = all_mins.min(axis=0)
    scene_max = all_maxs.max(axis=0)
    extent = scene_max - scene_min

    margin = capsule_radius
    x_range = np.linspace(scene_min[0] + margin, scene_max[0] - margin, grid_resolution)
    z_range = np.linspace(scene_min[2] + margin, scene_max[2] - margin, grid_resolution)
    xx, zz = np.meshgrid(x_range, z_range)
    grid_x = xx.ravel()
    grid_z = zz.ravel()

    ray_y = float(scene_max[1] + extent[1] * 0.1)
    heights = _ground_heights(grid_x, grid_z, ray_y, pf.hulls)

    total_cells = len(heights)
    ground_mask = np.isfinite(heights)
    ground_cells = int(ground_mask.sum())

    if ground_cells == 0:
        return SweepResult(
            grid_resolution=grid_resolution,
            total_cells=total_cells,
            ground_cells=0,
            connected_components=0,
            largest_component=0,
            traversability=0.0,
            island_sizes=[],
            seam_snags=[],
            capsule_radius=capsule_radius,
            capsule_height=capsule_height,
            step_height=step_height,
            scene_aabb_min=scene_min,
            scene_aabb_max=scene_max,
        )

    heights_grid = heights.reshape(grid_resolution, grid_resolution)
    ground_grid = ground_mask.reshape(grid_resolution, grid_resolution)

    adj = {}
    seam_snags = []

    for r in range(grid_resolution):
        for c in range(grid_resolution):
            if not ground_grid[r, c]:
                continue
            cell_id = r * grid_resolution + c
            if cell_id not in adj:
                adj[cell_id] = []

            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= grid_resolution or nc < 0 or nc >= grid_resolution:
                    continue
                if not ground_grid[nr, nc]:
                    continue

                neighbor_id = nr * grid_resolution + nc
                h_diff = abs(float(heights_grid[r, c] - heights_grid[nr, nc]))

                if h_diff > step_height:
                    snag_x = float(grid_x[cell_id] + grid_x[neighbor_id]) / 2
                    snag_y = float(max(heights_grid[r, c], heights_grid[nr, nc]))
                    snag_z = float(grid_z[cell_id] + grid_z[neighbor_id]) / 2
                    seam_snags.append((snag_x, snag_y, snag_z))
                    continue

                adj[cell_id].append(neighbor_id)
                if neighbor_id not in adj:
                    adj[neighbor_id] = []

    visited = set()
    components = []

    for cell_id in adj:
        if cell_id in visited:
            continue
        component = []
        stack = [cell_id]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for nb in adj.get(current, []):
                if nb not in visited:
                    stack.append(nb)
        components.append(len(component))

    components.sort(reverse=True)
    largest = components[0] if components else 0
    traversability = largest / ground_cells if ground_cells > 0 else 0.0

    unique_snags = dedup_snags(seam_snags, capsule_radius)

    return SweepResult(
        grid_resolution=grid_resolution,
        total_cells=total_cells,
        ground_cells=ground_cells,
        connected_components=len(components),
        largest_component=largest,
        traversability=traversability,
        island_sizes=components,
        seam_snags=unique_snags,
        capsule_radius=capsule_radius,
        capsule_height=capsule_height,
        step_height=step_height,
        scene_aabb_min=scene_min,
        scene_aabb_max=scene_max,
    )
