from __future__ import annotations

import numpy as np


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


def cull_occluded_hulls(hulls: list, viewpoints: int = 32) -> tuple[list, int]:
    """Remove hulls invisible from every exterior viewpoint.

    Hidden-point removal (Katz, Tal, Basri, "Direct Visibility of Point
    Sets", SIGGRAPH 2007; via Open3D) from fibonacci-sphere viewpoints
    over the union of hull surfaces. A hull none of whose surface samples
    is visible from any viewpoint is buried inside the union of its
    neighbors and can never be collided with from outside. Catches what
    single-hull containment culling misses: junk buried under several
    overlapping hulls with no one container.

    Not valid for environments, where the interior is reachable -- the
    caller is responsible for skipping those. Deterministic. Returns
    (kept_hulls, culled_count). No-op without open3d.
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

    kept = [h for i, h in enumerate(hulls) if visible[owners == i].any()]
    return kept, len(hulls) - len(kept)
