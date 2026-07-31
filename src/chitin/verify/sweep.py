from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chitin.phys import PhysHull, read_phys
from chitin.verify.raycast import ray_hull_spans
from chitin.verify.seam import dedup_snags

# Lateral samples around the capsule axis: the four axes plus the diagonals.
_RING_SAMPLES = 8
_EPS = 1e-4
# Span lookups are (columns x hulls); cap the block so scans with thousands of
# hulls stay bounded instead of allocating the whole product at once.
_SPAN_BLOCK = 2_000_000


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
    standable_cells: int = 0
    clearance_blocked: int = 0
    radius_blocked: int = 0

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
            "standable_cells": self.standable_cells,
            "clearance_blocked": self.clearance_blocked,
            "radius_blocked": self.radius_blocked,
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


def _as_phys_hulls(hulls: list) -> list[PhysHull]:
    out = []
    for h in hulls:
        if isinstance(h, PhysHull):
            out.append(h)
            continue
        verts = np.asarray(h.vertices, dtype=np.float32)
        out.append(
            PhysHull(
                vertices=verts,
                indices=h.indices,
                aabb_min=verts.min(axis=0),
                aabb_max=verts.max(axis=0),
            )
        )
    return out


def _solid_spans(
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    ray_y: float,
    hulls: list[PhysHull],
) -> tuple[np.ndarray, np.ndarray]:
    """Per column, the world-Y (bottom, top) of every hull the column crosses.

    Both arrays are ``(n_columns, n_hulls)`` and hold ``nan`` where the column
    misses the hull.
    """
    origins = np.stack([grid_x, np.full_like(grid_x, ray_y), grid_z], axis=1).astype(
        np.float32
    )
    direction = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    near, far = ray_hull_spans(origins, direction, hulls)

    tops = ray_y - near
    bottoms = ray_y - far
    miss = ~np.isfinite(near)
    tops[miss] = np.nan
    bottoms[miss] = np.nan
    return bottoms, tops


def _column_ground(
    bottoms_row: np.ndarray,
    tops_row: np.ndarray,
    capsule_height: float,
) -> tuple[float, bool]:
    """Lowest surface in one column carrying `capsule_height` of free space.

    Overlapping hulls are merged into single solid spans first, so a stack of
    decomposed hulls reads as one obstacle. Returns the surface height and
    whether a lower surface was skipped for insufficient headroom -- the
    topmost surface is always standable, since nothing is above it.
    """
    keep = np.isfinite(tops_row)
    if not keep.any():
        return float("nan"), False

    order = np.argsort(bottoms_row[keep])
    lo = bottoms_row[keep][order]
    hi = tops_row[keep][order]

    span_top = float(hi[0])
    climbed = False
    for i in range(1, len(lo)):
        if lo[i] <= span_top + _EPS:
            span_top = max(span_top, float(hi[i]))
            continue
        if lo[i] - span_top >= capsule_height:
            return span_top, climbed
        climbed = True
        span_top = float(hi[i])

    return span_top, climbed


def _block_columns(n_hulls: int) -> int:
    return max(1, _SPAN_BLOCK // max(1, n_hulls))


def _ground_layer(
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    ray_y: float,
    hulls: list[PhysHull],
    capsule_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground height, headroom-climb flag and hit flag for every column."""
    n_columns = len(grid_x)
    heights = np.full(n_columns, np.nan)
    climbed = np.zeros(n_columns, dtype=bool)
    hit = np.zeros(n_columns, dtype=bool)

    block = _block_columns(len(hulls))
    for start in range(0, n_columns, block):
        stop = min(start + block, n_columns)
        bottoms, tops = _solid_spans(
            grid_x[start:stop], grid_z[start:stop], ray_y, hulls
        )
        hit[start:stop] = np.isfinite(tops).any(axis=1)
        for i in range(stop - start):
            heights[start + i], climbed[start + i] = _column_ground(
                bottoms[i], tops[i], capsule_height
            )

    return heights, climbed, hit


def _radius_blocked(
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    heights: np.ndarray,
    ray_y: float,
    hulls: list[PhysHull],
    capsule_radius: float,
    capsule_height: float,
    step_height: float,
) -> np.ndarray:
    """Cells where the capsule's lateral ring intersects geometry.

    A ring sample blocks the cell when a solid span crosses the band between
    step height and head height above the cell's ground: anything lower is a
    step the capsule climbs, anything higher clears its head.
    """
    blocked = np.zeros(len(heights), dtype=bool)
    cells = np.where(np.isfinite(heights))[0]
    if len(cells) == 0 or capsule_radius <= 0 or capsule_height <= step_height:
        return blocked

    angles = np.linspace(0.0, 2.0 * np.pi, _RING_SAMPLES, endpoint=False)
    block = max(1, _block_columns(len(hulls)) // _RING_SAMPLES)

    for start in range(0, len(cells), block):
        sub = cells[start : start + block]
        ring_x = (grid_x[sub][:, None] + np.cos(angles) * capsule_radius).ravel()
        ring_z = (grid_z[sub][:, None] + np.sin(angles) * capsule_radius).ravel()

        bottoms, tops = _solid_spans(ring_x, ring_z, ray_y, hulls)
        floor = np.repeat(heights[sub], _RING_SAMPLES)
        band_low = (floor + step_height)[:, None]
        band_high = (floor + capsule_height)[:, None]

        overlaps = (
            np.isfinite(tops) & (tops > band_low + _EPS) & (bottoms < band_high - _EPS)
        )
        blocked[sub] = overlaps.any(axis=1).reshape(-1, _RING_SAMPLES).any(axis=1)

    return blocked


def _empty_result(
    grid_resolution: int,
    total_cells: int,
    ground_cells: int,
    capsule_radius: float,
    capsule_height: float,
    step_height: float,
    scene_min: np.ndarray,
    scene_max: np.ndarray,
    clearance_blocked: int = 0,
    radius_blocked: int = 0,
) -> SweepResult:
    return SweepResult(
        grid_resolution=grid_resolution,
        total_cells=total_cells,
        ground_cells=ground_cells,
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
        standable_cells=0,
        clearance_blocked=clearance_blocked,
        radius_blocked=radius_blocked,
    )


def sweep_hulls(
    hulls: list,
    grid_resolution: int = 32,
    capsule_radius: float = 0.3,
    capsule_height: float = 1.8,
    step_height: float = 0.3,
) -> SweepResult:
    """Capsule traversability over a hull set.

    Each grid column is resolved to the lowest surface with `capsule_height` of
    free space above it, cells whose lateral ring hits geometry are dropped,
    and the survivors are flood-filled with `step_height`-gated adjacency.
    """
    total_cells = grid_resolution * grid_resolution

    if not hulls:
        return _empty_result(
            grid_resolution,
            0,
            0,
            capsule_radius,
            capsule_height,
            step_height,
            np.zeros(3),
            np.zeros(3),
        )

    phys_hulls = _as_phys_hulls(hulls)
    all_mins = np.array([h.aabb_min for h in phys_hulls])
    all_maxs = np.array([h.aabb_max for h in phys_hulls])
    scene_min = all_mins.min(axis=0)
    scene_max = all_maxs.max(axis=0)
    extent = scene_max - scene_min

    margin = capsule_radius
    x_range = np.linspace(scene_min[0] + margin, scene_max[0] - margin, grid_resolution)
    z_range = np.linspace(scene_min[2] + margin, scene_max[2] - margin, grid_resolution)
    xx, zz = np.meshgrid(x_range, z_range)
    grid_x = xx.ravel()
    grid_z = zz.ravel()

    ray_y = float(scene_max[1] + extent[1] * 0.1 + 1.0)
    heights, climbed, hit = _ground_layer(
        grid_x, grid_z, ray_y, phys_hulls, capsule_height
    )
    ground_cells = int(hit.sum())
    clearance_blocked = int(climbed.sum())

    ring_blocked = _radius_blocked(
        grid_x,
        grid_z,
        heights,
        ray_y,
        phys_hulls,
        capsule_radius,
        capsule_height,
        step_height,
    )
    radius_blocked = int(ring_blocked.sum())
    heights[ring_blocked] = np.nan

    standable_mask = np.isfinite(heights)
    standable_cells = int(standable_mask.sum())

    if standable_cells == 0:
        return _empty_result(
            grid_resolution,
            total_cells,
            ground_cells,
            capsule_radius,
            capsule_height,
            step_height,
            scene_min,
            scene_max,
            clearance_blocked,
            radius_blocked,
        )

    heights_grid = heights.reshape(grid_resolution, grid_resolution)
    ground_grid = standable_mask.reshape(grid_resolution, grid_resolution)

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
    traversability = largest / standable_cells

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
        standable_cells=standable_cells,
        clearance_blocked=clearance_blocked,
        radius_blocked=radius_blocked,
    )


def sweep(
    phys_path: str | Path,
    grid_resolution: int = 32,
    capsule_radius: float = 0.3,
    capsule_height: float = 1.8,
    step_height: float = 0.3,
) -> SweepResult:
    pf = read_phys(phys_path)
    return sweep_hulls(
        pf.hulls,
        grid_resolution=grid_resolution,
        capsule_radius=capsule_radius,
        capsule_height=capsule_height,
        step_height=step_height,
    )
