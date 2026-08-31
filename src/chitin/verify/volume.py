"""Volume-based collider quality measurement.

Estimates what fraction of collider-occupied space is genuine source volume
versus false fill (phantom collision). Matches the deterministic Halton-based
algorithm in @autarkis/chitin-lite's quality.ts (``evaluateColliderQuality``,
non-per-component branch).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chitin._metric_names import (
    COLLIDER_VOLUME_PRECISION,
    DEEP_FALSE_FILL_FRACTION,
    FALSE_FILL_FRACTION,
    QUALITY_METHOD,
    QUALITY_VOLUME_SAMPLES,
)
from chitin.verify.convex import outward_face_planes, points_inside

DEFAULT_VOLUME_SAMPLES = 4096
DEFAULT_MIN_COLLIDER_SAMPLES = 32
DEFAULT_DEEP_FILL_CLEARANCE_RATIO = 0.02

# Point-triangle pair budget per chunk, bounding peak memory for the
# vectorized ray/distance kernels below.
_CHUNK_PAIR_BUDGET = 20_000_000

# Fixed ray direction for the parity test, matching quality.ts's RAY_DIRECTION.
_RAY_DIRECTION = np.array([1.0, 0.372013, 0.529117], dtype=np.float64)
_RAY_DIRECTION /= np.linalg.norm(_RAY_DIRECTION)


@dataclass(frozen=True)
class VolumeResult:
    collider_volume_precision: float | None
    false_fill_fraction: float | None
    deep_false_fill_fraction: float | None
    volume_samples: int
    collider_volume_samples: int

    def to_coverage_dict(self) -> dict:
        """Merge into a coverage report dict."""
        return {
            COLLIDER_VOLUME_PRECISION: self.collider_volume_precision,
            FALSE_FILL_FRACTION: self.false_fill_fraction,
            DEEP_FALSE_FILL_FRACTION: self.deep_false_fill_fraction,
            QUALITY_METHOD: "deterministic_halton_v1",
            QUALITY_VOLUME_SAMPLES: self.volume_samples,
        }


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    fraction = 1.0 / base
    while index > 0:
        result += (index % base) * fraction
        index //= base
        fraction /= base
    return result


def _halton_samples(
    n: int, bounds_min: np.ndarray, bounds_max: np.ndarray, start: int = 1
) -> np.ndarray:
    """Halton(2,3,5) points mapped into the [bounds_min, bounds_max] AABB."""
    unit = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        index = start + i
        unit[i, 0] = _radical_inverse(index, 2)
        unit[i, 1] = _radical_inverse(index, 3)
        unit[i, 2] = _radical_inverse(index, 5)
    extent = np.asarray(bounds_max, dtype=np.float64) - np.asarray(
        bounds_min, dtype=np.float64
    )
    return np.asarray(bounds_min, dtype=np.float64) + unit * extent


def _prepared_hulls(hulls: list) -> list:
    """Hulls with at least one non-degenerate triangle."""
    prepared = []
    for hull in hulls:
        faces = np.asarray(hull.indices).reshape(-1, 3)
        if faces.shape[0] > 0:
            prepared.append(hull)
    return prepared


def _combined_hull_bounds(
    hulls: list, fallback_min: np.ndarray, fallback_max: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not hulls:
        return np.asarray(fallback_min, dtype=np.float64), np.asarray(
            fallback_max, dtype=np.float64
        )
    mins = np.stack([hull.vertices.astype(np.float64).min(axis=0) for hull in hulls])
    maxs = np.stack([hull.vertices.astype(np.float64).max(axis=0) for hull in hulls])
    return mins.min(axis=0), maxs.max(axis=0)


def _inside_any_hull(points: np.ndarray, hulls: list, tol: float) -> np.ndarray:
    """For each point, is it inside at least one (already-prepared) hull."""
    inside = np.zeros(len(points), dtype=bool)
    for hull in hulls:
        normals, d = outward_face_planes(hull)
        inside |= points_inside(normals, d, points, tol)
    return inside


def _chunk_size_for(n_points: int, n_faces: int) -> int:
    if n_faces <= 0:
        return max(1, n_points)
    size = _CHUNK_PAIR_BUDGET // n_faces
    return max(1, min(n_points, size))


def _ray_mesh_parity(
    points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, tol: float
) -> np.ndarray:
    """Möller-Trumbore ray/triangle parity test: odd hit count == inside.

    ``points`` are ray origins; the ray direction is the fixed, arbitrary
    ``_RAY_DIRECTION`` (chosen to make axis-aligned degeneracies unlikely).
    Nearly-coincident hits (shared edges/vertices) within ``tol`` of each
    other are deduplicated before counting parity, matching quality.ts.
    """
    n = len(points)
    result = np.zeros(n, dtype=bool)
    if n == 0 or len(faces) == 0:
        return result

    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    edge1 = b - a
    edge2 = c - a
    ray_dir = _RAY_DIRECTION

    pvec = np.cross(ray_dir, edge2)  # (F, 3)
    det = np.einsum("fi,fi->f", edge1, pvec)  # (F,)
    invalid_det = np.abs(det) < 1e-12
    det_safe = np.where(invalid_det, 1.0, det)
    inv_det = 1.0 / det_safe

    chunk = _chunk_size_for(n, len(faces))
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        pts = points[start:end]  # (P, 3)

        tvec = pts[:, None, :] - a[None, :, :]  # (P, F, 3)
        u = np.einsum("pfi,fi->pf", tvec, pvec) * inv_det[None, :]
        qvec = np.cross(tvec, edge1[None, :, :])  # (P, F, 3)
        v = np.einsum("pfi,i->pf", qvec, ray_dir) * inv_det[None, :]
        t = np.einsum("fi,pfi->pf", edge2, qvec) * inv_det[None, :]

        valid = (
            (~invalid_det[None, :])
            & (u >= 0)
            & (u <= 1)
            & (v >= 0)
            & (u + v <= 1)
            & (t > 0)
        )

        t_masked = np.where(valid, t, np.inf)
        t_sorted = np.sort(t_masked, axis=1)
        diffs = np.diff(t_sorted, axis=1, prepend=-np.inf)
        finite = np.isfinite(t_sorted)
        is_new_hit = (diffs > tol) & finite
        counts = is_new_hit.sum(axis=1)
        result[start:end] = (counts % 2) == 1

    return result


def _point_triangle_distance_squared_chunk(
    points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    ab: np.ndarray,
    ac: np.ndarray,
    bc: np.ndarray,
) -> np.ndarray:
    """Closest-point-on-triangle squared distance (Ericson's algorithm).

    ``points`` is (P, 3); ``a``/``b``/``c``/``ab``/``ac``/``bc`` are (F, 3).
    Returns (P, F) squared distances.
    """
    ap = points[:, None, :] - a[None, :, :]  # (P, F, 3)
    d1 = np.einsum("fi,pfi->pf", ab, ap)
    d2 = np.einsum("fi,pfi->pf", ac, ap)
    case1 = (d1 <= 0) & (d2 <= 0)

    bp = points[:, None, :] - b[None, :, :]
    d3 = np.einsum("fi,pfi->pf", ab, bp)
    d4 = np.einsum("fi,pfi->pf", ac, bp)
    case2 = (~case1) & (d3 >= 0) & (d4 <= d3)

    vc = d1 * d4 - d3 * d2
    case3 = (~case1) & (~case2) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)

    cp = points[:, None, :] - c[None, :, :]
    d5 = np.einsum("fi,pfi->pf", ab, cp)
    d6 = np.einsum("fi,pfi->pf", ac, cp)
    case4 = (~case1) & (~case2) & (~case3) & (d6 >= 0) & (d5 <= d6)

    vb = d5 * d2 - d1 * d6
    case5 = (
        (~case1) & (~case2) & (~case3) & (~case4) & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    )

    va = d3 * d6 - d5 * d4
    remaining = ~(case1 | case2 | case3 | case4 | case5)
    case6 = remaining & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    case7 = remaining & (~case6)

    out = np.empty(case1.shape, dtype=np.float64)

    out[case1] = np.sum(ap[case1] ** 2, axis=-1)
    out[case2] = np.sum(bp[case2] ** 2, axis=-1)

    if np.any(case3):
        denom3 = d1 - d3
        v3 = np.where(denom3 != 0, d1 / np.where(denom3 == 0, 1.0, denom3), 0.0)
        proj3 = a[None, :, :] + v3[..., None] * ab[None, :, :]
        diff3 = points[:, None, :] - proj3
        out[case3] = np.sum(diff3[case3] ** 2, axis=-1)

    out[case4] = np.sum(cp[case4] ** 2, axis=-1)

    if np.any(case5):
        denom5 = d2 - d6
        w5 = np.where(denom5 != 0, d2 / np.where(denom5 == 0, 1.0, denom5), 0.0)
        proj5 = a[None, :, :] + w5[..., None] * ac[None, :, :]
        diff5 = points[:, None, :] - proj5
        out[case5] = np.sum(diff5[case5] ** 2, axis=-1)

    if np.any(case6):
        denom6 = (d4 - d3) + (d5 - d6)
        w6 = np.where(denom6 != 0, (d4 - d3) / np.where(denom6 == 0, 1.0, denom6), 0.0)
        proj6 = b[None, :, :] + w6[..., None] * bc[None, :, :]
        diff6 = points[:, None, :] - proj6
        out[case6] = np.sum(diff6[case6] ** 2, axis=-1)

    if np.any(case7):
        denom7 = va + vb + vc
        denom7_safe = np.where(denom7 != 0, denom7, 1.0)
        v7 = vb / denom7_safe
        w7 = vc / denom7_safe
        proj7 = (
            a[None, :, :]
            + ab[None, :, :] * v7[..., None]
            + ac[None, :, :] * w7[..., None]
        )
        diff7 = points[:, None, :] - proj7
        out[case7] = np.sum(diff7[case7] ** 2, axis=-1)

    return out.min(axis=1)


def _point_mesh_distance_squared(
    points: np.ndarray, vertices: np.ndarray, faces: np.ndarray
) -> np.ndarray:
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if len(faces) == 0:
        return np.full(n, np.inf, dtype=np.float64)

    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    ab = b - a
    ac = c - a
    bc = c - b

    result = np.empty(n, dtype=np.float64)
    chunk = _chunk_size_for(n, len(faces))
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        result[start:end] = _point_triangle_distance_squared_chunk(
            points[start:end], a, b, c, ab, ac, bc
        )
    return result


def volume_report(
    hulls: list,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    *,
    volume_samples: int = DEFAULT_VOLUME_SAMPLES,
    min_collider_samples: int = DEFAULT_MIN_COLLIDER_SAMPLES,
    deep_fill_clearance_ratio: float = DEFAULT_DEEP_FILL_CLEARANCE_RATIO,
    tolerance: float | None = None,
) -> VolumeResult:
    """Estimate what fraction of collider-occupied volume is real source volume.

    Deterministically samples the collider AABB with a 3D Halton(2,3,5)
    sequence. Samples that land inside the hull set are classified as true
    fill (inside the source mesh, via a ray-parity test) or false fill;
    false-fill samples farther than ``deep_fill_clearance_ratio`` * source
    diagonal from the source surface are additionally flagged as deep false
    fill (severe phantom collision, not just surface slop).

    Returns ``None`` for all fractions when fewer than ``min_collider_samples``
    samples land inside the collider (too little signal to estimate).
    """
    if volume_samples < 1:
        raise ValueError(
            f"volume_samples must be a positive integer, got {volume_samples}"
        )
    if min_collider_samples < 1:
        raise ValueError(
            f"min_collider_samples must be a positive integer, got {min_collider_samples}"
        )
    if not (0.0 <= deep_fill_clearance_ratio <= 1.0):
        raise ValueError(
            "deep_fill_clearance_ratio must be in [0, 1], got "
            f"{deep_fill_clearance_ratio}"
        )

    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    source_faces = np.asarray(source_faces).reshape(-1, 3).astype(np.int64)

    source_min = source_vertices.min(axis=0)
    source_max = source_vertices.max(axis=0)
    source_diagonal = float(np.linalg.norm(source_max - source_min))
    tol = tolerance if tolerance is not None else max(source_diagonal * 1e-5, 1e-9)

    prepared_hulls = _prepared_hulls(hulls)
    collider_min, collider_max = _combined_hull_bounds(
        prepared_hulls, source_min, source_max
    )

    samples = _halton_samples(volume_samples, collider_min, collider_max, start=1)

    inside_hull_mask = _inside_any_hull(samples, prepared_hulls, tol)
    collider_points = samples[inside_hull_mask]
    collider_volume_samples = int(inside_hull_mask.sum())

    true_samples = 0
    deep_false_fill_samples = 0
    if collider_volume_samples > 0:
        inside_source_mask = _ray_mesh_parity(
            collider_points, source_vertices, source_faces, tol
        )
        true_samples = int(inside_source_mask.sum())

        outside_points = collider_points[~inside_source_mask]
        if len(outside_points) > 0:
            distance_sq = _point_mesh_distance_squared(
                outside_points, source_vertices, source_faces
            )
            clearance_sq = (source_diagonal * deep_fill_clearance_ratio) ** 2
            deep_false_fill_samples = int(np.sum(distance_sq > clearance_sq))

    has_signal = collider_volume_samples >= min_collider_samples
    precision = true_samples / collider_volume_samples if has_signal else None
    false_fill = (1.0 - precision) if precision is not None else None
    deep_false_fill = (
        deep_false_fill_samples / collider_volume_samples if has_signal else None
    )

    return VolumeResult(
        collider_volume_precision=precision,
        false_fill_fraction=false_fill,
        deep_false_fill_fraction=deep_false_fill,
        volume_samples=volume_samples,
        collider_volume_samples=collider_volume_samples,
    )
