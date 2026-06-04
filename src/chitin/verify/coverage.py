from __future__ import annotations

import numpy as np

from chitin.verify.convex import outward_face_planes, point_plane_margins

MAX_COVERAGE_SAMPLES = 200_000


def coverage_report(
    hulls: list,
    points: np.ndarray,
    cell_indices: list[np.ndarray] | None = None,
    tol_fraction: float = 0.001,
    max_samples: int = MAX_COVERAGE_SAMPLES,
    seed: int = 0,
) -> dict:
    """Classify input samples as covered/uncovered by the hull set.

    The input points are ground truth: every sample should sit inside (or
    within tolerance of) at least one hull. Returns a JSON-ready dict with
    the global covered fraction, slack percentiles over covered points
    (depth inside the deepest containing hull), and, when ``cell_indices``
    is given, per-cell fractions with the worst decile called out — means
    hide holes.

    Tolerance is ``tol_fraction`` of the scene diagonal: reconstruction
    smooths the surface, so raw samples sit slightly off the hull faces.
    Sampling above ``max_samples`` is deterministic for a given ``seed``.
    """
    n_input = len(points)
    if n_input == 0:
        return {"input_count": 0, "sample_count": 0, "covered_fraction": 0.0}

    if n_input > max_samples:
        rng = np.random.default_rng(seed)
        sample_idx = np.sort(rng.choice(n_input, size=max_samples, replace=False))
    else:
        sample_idx = np.arange(n_input)
    pts = np.asarray(points, dtype=np.float64)[sample_idx]

    scene_min = pts.min(axis=0)
    scene_max = pts.max(axis=0)
    diagonal = float(np.linalg.norm(scene_max - scene_min))
    tol = tol_fraction * diagonal

    covered = np.zeros(len(pts), dtype=bool)
    slack = np.full(len(pts), -np.inf)

    for hull in hulls:
        h_min = hull.vertices.min(axis=0) - tol
        h_max = hull.vertices.max(axis=0) + tol
        in_aabb = np.all((pts >= h_min) & (pts <= h_max), axis=1)
        if not np.any(in_aabb):
            continue
        normals, d = outward_face_planes(hull)
        margins = point_plane_margins(normals, d, pts[in_aabb])
        inside = margins >= -tol
        if not np.any(inside):
            continue
        idx = np.where(in_aabb)[0][inside]
        covered[idx] = True
        slack[idx] = np.maximum(slack[idx], margins[inside])

    report: dict = {
        "input_count": n_input,
        "sample_count": int(len(pts)),
        "tolerance": round(tol, 6),
        "covered_fraction": round(float(covered.mean()), 4),
        "uncovered_count": int((~covered).sum()),
    }

    if np.any(covered):
        covered_slack = slack[covered]
        report["slack_p50"] = round(float(np.percentile(covered_slack, 50)), 6)
        report["slack_p95"] = round(float(np.percentile(covered_slack, 95)), 6)

    if cell_indices:
        sampled_pos = np.full(n_input, -1, dtype=np.int64)
        sampled_pos[sample_idx] = np.arange(len(sample_idx))
        fractions = []
        for cell_id, indices in enumerate(cell_indices):
            local = sampled_pos[indices]
            local = local[local >= 0]
            if len(local) == 0:
                continue
            fractions.append((cell_id, float(covered[local].mean()), len(local)))
        if fractions:
            fractions.sort(key=lambda f: f[1])
            decile = max(1, len(fractions) // 10)
            report["cell_count"] = len(fractions)
            report["worst_cell_fraction"] = round(fractions[0][1], 4)
            report["worst_decile_fraction"] = round(
                float(np.mean([f[1] for f in fractions[:decile]])), 4
            )
            report["worst_cells"] = [
                {"cell": cid, "covered_fraction": round(frac, 4), "samples": cnt}
                for cid, frac, cnt in fractions[:10]
            ]

    return report
