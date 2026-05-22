from __future__ import annotations

import numpy as np

from chitin.phys import PhysHull


def _moller_trumbore(
    cand_origins: np.ndarray,
    direction: np.ndarray,
    hull: PhysHull,
) -> tuple[np.ndarray, np.ndarray]:
    idx = hull.indices.reshape(-1, 3)
    v0 = hull.vertices[idx[:, 0]]
    e1 = hull.vertices[idx[:, 1]] - v0
    e2 = hull.vertices[idx[:, 2]] - v0

    h = np.cross(direction, e2)
    a = np.einsum("ij,ij->i", e1, h)
    valid = np.abs(a) > 1e-10
    inv_a = np.zeros_like(a)
    inv_a[valid] = 1.0 / a[valid]

    s = cand_origins[:, np.newaxis, :] - v0[np.newaxis, :, :]
    u = np.einsum("rij,ij->ri", s, h) * inv_a[np.newaxis, :]
    ok = valid[np.newaxis, :] & (u >= 0) & (u <= 1)
    q = np.cross(s, e1[np.newaxis, :, :])
    v = np.einsum("rij,j->ri", q, direction) * inv_a[np.newaxis, :]
    ok &= (v >= 0) & (u + v <= 1)
    t = np.einsum("rij,ij->ri", q, e2) * inv_a[np.newaxis, :]
    ok &= t > 1e-6
    return ok, t


def _xz_candidates(
    origins: np.ndarray, hull: PhysHull, exclude: np.ndarray | None = None
) -> np.ndarray:
    hmin, hmax = hull.aabb_min, hull.aabb_max
    in_xz = (
        (origins[:, 0] >= hmin[0])
        & (origins[:, 0] <= hmax[0])
        & (origins[:, 2] >= hmin[2])
        & (origins[:, 2] <= hmax[2])
    )
    if exclude is not None:
        in_xz &= ~exclude
    return np.where(in_xz)[0]


def ray_hits_any(
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
        candidates = _xz_candidates(origins, hull, exclude=hit)
        if len(candidates) == 0:
            continue

        ok, _t = _moller_trumbore(origins[candidates], direction, hull)
        cand_hits = np.any(ok, axis=1)
        hit[candidates[cand_hits]] = True

    return hit


def ray_closest_hit(
    origins: np.ndarray,
    direction: np.ndarray,
    hulls: list[PhysHull],
) -> np.ndarray:
    n_rays = len(origins)
    closest_t = np.full(n_rays, np.inf, dtype=np.float64)

    for hull in hulls:
        candidates = _xz_candidates(origins, hull)
        if len(candidates) == 0:
            continue

        ok, t = _moller_trumbore(origins[candidates], direction, hull)
        t_masked = np.where(ok, t, np.inf)
        min_t_per_ray = t_masked.min(axis=1)
        np.minimum.at(closest_t, candidates, min_t_per_ray)

    return closest_t
