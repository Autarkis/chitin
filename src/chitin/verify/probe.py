# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chitin.phys import read_phys
from chitin.verify.raycast import ray_hits_any


@dataclass
class ProbeResult:
    grid_resolution: int
    total_rays: int
    hits: int
    misses: int
    coverage: float
    gap_positions: np.ndarray
    scene_aabb_min: np.ndarray
    scene_aabb_max: np.ndarray
    capsule_radius: float
    gap_clusters: int

    @property
    def confidence(self) -> str:
        if self.coverage >= 0.95:
            return "high"
        if self.coverage >= 0.80:
            return "medium"
        return "low"

    def to_json(self, path: str | Path) -> None:
        data = {
            "grid_resolution": self.grid_resolution,
            "total_rays": self.total_rays,
            "hits": self.hits,
            "misses": self.misses,
            "coverage": round(self.coverage, 4),
            "confidence": self.confidence,
            "capsule_radius": self.capsule_radius,
            "gap_clusters": self.gap_clusters,
            "scene_aabb_min": self.scene_aabb_min.tolist(),
            "scene_aabb_max": self.scene_aabb_max.tolist(),
            "gap_positions": self.gap_positions.tolist(),
        }
        Path(path).write_text(json.dumps(data, indent=2))


def _cluster_gaps(gap_positions: np.ndarray, capsule_radius: float) -> int:
    if len(gap_positions) == 0:
        return 0
    remaining = set(range(len(gap_positions)))
    clusters = 0
    while remaining:
        clusters += 1
        seed = remaining.pop()
        stack = [seed]
        while stack:
            current = stack.pop()
            pt = gap_positions[current]
            to_remove = []
            for idx in remaining:
                if np.linalg.norm(gap_positions[idx] - pt) <= capsule_radius * 2:
                    to_remove.append(idx)
            for idx in to_remove:
                remaining.discard(idx)
                stack.append(idx)
    return clusters


def probe(
    phys_path: str | Path,
    grid_resolution: int = 64,
    capsule_radius: float = 0.3,
) -> ProbeResult:
    pf = read_phys(phys_path)

    if not pf.hulls:
        return ProbeResult(
            grid_resolution=grid_resolution,
            total_rays=0,
            hits=0,
            misses=0,
            coverage=0.0,
            gap_positions=np.empty((0, 2), dtype=np.float32),
            scene_aabb_min=np.zeros(3),
            scene_aabb_max=np.zeros(3),
            capsule_radius=capsule_radius,
            gap_clusters=0,
        )

    all_mins = np.array([h.aabb_min for h in pf.hulls])
    all_maxs = np.array([h.aabb_max for h in pf.hulls])
    scene_min = all_mins.min(axis=0)
    scene_max = all_maxs.max(axis=0)

    extent = scene_max - scene_min
    x_range = np.linspace(scene_min[0], scene_max[0], grid_resolution)
    z_range = np.linspace(scene_min[2], scene_max[2], grid_resolution)
    xx, zz = np.meshgrid(x_range, z_range)
    ray_x = xx.ravel()
    ray_z = zz.ravel()
    ray_y = np.full_like(ray_x, scene_max[1] + extent[1] * 0.1)

    origins = np.stack([ray_x, ray_y, ray_z], axis=1).astype(np.float32)
    direction = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    hits = ray_hits_any(origins, direction, pf.hulls)

    hit_count = int(hits.sum())
    miss_count = len(hits) - hit_count
    coverage = hit_count / len(hits) if len(hits) > 0 else 0.0

    gap_positions = origins[~hits][:, [0, 2]]
    gap_clusters = _cluster_gaps(gap_positions, capsule_radius)

    return ProbeResult(
        grid_resolution=grid_resolution,
        total_rays=len(origins),
        hits=hit_count,
        misses=miss_count,
        coverage=coverage,
        gap_positions=gap_positions,
        scene_aabb_min=scene_min,
        scene_aabb_max=scene_max,
        capsule_radius=capsule_radius,
        gap_clusters=gap_clusters,
    )
