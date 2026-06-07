from __future__ import annotations

import numpy as np

from chitin.verify.convex import outward_face_planes, point_plane_margins
from chitin.verify.coverage import MAX_COVERAGE_SAMPLES


def _fibonacci_directions(count: int) -> np.ndarray:
    i = np.arange(count, dtype=np.float64) + 0.5
    phi = np.arccos(1 - 2 * i / count)
    theta = np.pi * (1 + 5**0.5) * i
    return np.column_stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    )


# Barycentric grid weights (n=4) used to sample each face interior.
# HPR occlusion needs sampling spacing finer than the buried-to-occluder
# gap; vertices plus face centroids alone are too sparse to occlude.
_BARY_DIVISIONS = 4
_BARY_WEIGHTS = np.array(
    [
        [
            i / _BARY_DIVISIONS,
            j / _BARY_DIVISIONS,
            (_BARY_DIVISIONS - i - j) / _BARY_DIVISIONS,
        ]
        for i in range(_BARY_DIVISIONS + 1)
        for j in range(_BARY_DIVISIONS + 1 - i)
    ],
    dtype=np.float64,
)


def _hull_surface_samples(hull) -> np.ndarray:
    verts = hull.vertices.astype(np.float64)
    faces = hull.indices.reshape(-1, 3)
    tris = verts[faces]
    face_samples = np.einsum("bk,fkd->fbd", _BARY_WEIGHTS, tris).reshape(-1, 3)
    return np.vstack([verts, face_samples])


def cull_occluded_hulls(
    hulls: list,
    points: np.ndarray | None = None,
    viewpoints: int = 32,
    tol_fraction: float = 0.001,
) -> tuple[list, int]:
    """Remove hulls invisible from every exterior viewpoint.

    Hidden-point removal (Katz, Tal, Basri, "Direct Visibility of Point
    Sets", SIGGRAPH 2007; via Open3D) from fibonacci-sphere viewpoints
    over the union of hull surfaces. A hull none of whose surface samples
    is visible from any viewpoint is buried inside the union of its
    neighbors and can never be collided with from outside. Catches what
    single-hull containment culling misses: junk buried under several
    overlapping hulls with no one container.

    Exterior visibility alone is not a valid cull for environments, where
    the interior is reachable. Passing the input ``points`` makes the cull
    coverage-guarded: an invisible hull is removed only if every input
    point it covers (within ``tol_fraction`` of the scene diagonal, the
    same tolerance as ``coverage_report``) stays covered by the remaining
    hulls. Without ``points`` the caller is responsible for skipping
    environments. Deterministic. Returns (kept_hulls, culled_count).
    No-op without open3d.
    """
    if len(hulls) <= 1:
        return hulls, 0
    try:
        import open3d as o3d
    except ImportError:
        return hulls, 0

    samples = [_hull_surface_samples(h) for h in hulls]
    owners = np.concatenate(
        [np.full(len(s), i, dtype=np.int64) for i, s in enumerate(samples)]
    )
    cloud = np.vstack(samples)

    bounds_min = cloud.min(axis=0)
    bounds_max = cloud.max(axis=0)
    extent = float(np.linalg.norm(bounds_max - bounds_min))
    if extent <= 0:
        return hulls, 0
    center = 0.5 * (bounds_min + bounds_max)
    camera_radius = 1.5 * extent
    # HPR spherical-flip radius. The customary 100x is tuned for dense
    # scan clouds and leaks false-visibles on hull-surface sampling
    # (buried boxes report ~10% visible); 10x separates cleanly while
    # separated hulls remain fully visible down to 1x. A hull is kept if
    # ANY sample is visible from ANY viewpoint, so this stays
    # conservative for concave pockets.
    hpr_radius = 10.0 * extent

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)

    visible = np.zeros(len(cloud), dtype=bool)
    for direction in _fibonacci_directions(viewpoints):
        camera = center + camera_radius * direction
        _, idx = pcd.hidden_point_removal(camera.tolist(), hpr_radius)
        visible[np.asarray(idx, dtype=np.int64)] = True

    occluded = [i for i in range(len(hulls)) if not visible[owners == i].any()]
    if occluded and points is not None and len(points) > 0:
        occluded = _coverage_safe_culls(hulls, occluded, points, tol_fraction)
    culled = set(occluded)
    kept = [h for i, h in enumerate(hulls) if i not in culled]
    return kept, len(hulls) - len(kept)


def _covered_indices(hull, points: np.ndarray, tol: float) -> np.ndarray:
    h_min = hull.vertices.min(axis=0) - tol
    h_max = hull.vertices.max(axis=0) + tol
    in_aabb = np.all((points >= h_min) & (points <= h_max), axis=1)
    if not np.any(in_aabb):
        return np.empty(0, dtype=np.int64)
    normals, d = outward_face_planes(hull)
    inside = point_plane_margins(normals, d, points[in_aabb]) >= -tol
    return np.where(in_aabb)[0][inside]


def _coverage_safe_culls(
    hulls: list, occluded: list[int], points: np.ndarray, tol_fraction: float
) -> list[int]:
    """Filter ``occluded`` down to hulls whose removal cannot uncover points.

    A candidate is cullable only when every input point it covers is also
    covered by a non-candidate hull or an already-kept candidate.
    Candidates are decided in hull order, so of two invisible hulls
    covering the same otherwise-orphaned points the first is kept and the
    second can still go.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) > MAX_COVERAGE_SAMPLES:
        rng = np.random.default_rng(0)
        choice = rng.choice(len(points), size=MAX_COVERAGE_SAMPLES, replace=False)
        points = points[np.sort(choice)]

    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    tol = tol_fraction * diagonal

    cand_cover = {i: _covered_indices(hulls[i], points, tol) for i in occluded}
    union = np.unique(np.concatenate(list(cand_cover.values())))
    if union.size == 0:
        return occluded

    sub = points[union]
    sub_pos = np.full(len(points), -1, dtype=np.int64)
    sub_pos[union] = np.arange(union.size)

    covered = np.zeros(union.size, dtype=bool)
    occluded_set = set(occluded)
    for j, hull in enumerate(hulls):
        if j not in occluded_set:
            covered[_covered_indices(hull, sub, tol)] = True

    cullable = []
    for i in occluded:
        mine = sub_pos[cand_cover[i]]
        if covered[mine].all():
            cullable.append(i)
        else:
            covered[mine] = True
    return cullable
