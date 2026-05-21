# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chitin.phys import PhysHull, read_phys


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


def _ray_hits_any_hull(
    origins: np.ndarray,
    direction: np.ndarray,
    hulls: list[PhysHull],
) -> np.ndarray:
    n_rays = len(origins)
    hit = np.zeros(n_rays, dtype=bool)
    if not hulls:
        return hit

    for hull in hulls:
        if hit.all():
            break
        hmin, hmax = hull.aabb_min, hull.aabb_max
        in_xz = (
            (origins[:, 0] >= hmin[0])
            & (origins[:, 0] <= hmax[0])
            & (origins[:, 2] >= hmin[2])
            & (origins[:, 2] <= hmax[2])
        )
        candidates = np.where(in_xz & ~hit)[0]
        if len(candidates) == 0:
            continue

        idx = hull.indices.reshape(-1, 3)
        v0 = hull.vertices[idx[:, 0]]
        e1 = hull.vertices[idx[:, 1]] - v0
        e2 = hull.vertices[idx[:, 2]] - v0

        h = np.cross(direction, e2)
        a = np.einsum("ij,ij->i", e1, h)
        valid = np.abs(a) > 1e-10
        inv_a = np.zeros_like(a)
        inv_a[valid] = 1.0 / a[valid]

        cand_origins = origins[candidates]
        s = cand_origins[:, np.newaxis, :] - v0[np.newaxis, :, :]
        u = np.einsum("rij,ij->ri", s, h) * inv_a[np.newaxis, :]
        ok = valid[np.newaxis, :] & (u >= 0) & (u <= 1)
        q = np.cross(s, e1[np.newaxis, :, :])
        v = np.einsum("rij,j->ri", q, direction) * inv_a[np.newaxis, :]
        ok &= (v >= 0) & (u + v <= 1)
        t = np.einsum("rij,ij->ri", q, e2) * inv_a[np.newaxis, :]
        ok &= t > 1e-6
        cand_hits = np.any(ok, axis=1)
        hit[candidates[cand_hits]] = True

    return hit


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

    hits = _ray_hits_any_hull(origins, direction, pf.hulls)

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
