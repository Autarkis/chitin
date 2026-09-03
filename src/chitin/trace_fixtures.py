"""License-clean procedural fixtures for reference traces.

Each fixture is a pure-numpy mesh designed to exercise a specific
decomposition scenario. No external assets, no license concerns.
"""

from __future__ import annotations

import numpy as np


def box_mesh(
    extents: tuple[float, float, float] = (2.0, 2.0, 2.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box. Convex baseline — CoACD should return ~1 hull."""
    hx, hy, hz = [e / 2 for e in extents]
    vertices = np.array(
        [
            [-hx, -hy, -hz],
            [-hx, -hy, hz],
            [-hx, hy, -hz],
            [-hx, hy, hz],
            [hx, -hy, -hz],
            [hx, -hy, hz],
            [hx, hy, -hz],
            [hx, hy, hz],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def l_shape_mesh() -> tuple[np.ndarray, np.ndarray]:
    """L-shaped concave mesh. Forces multi-hull decomposition."""
    vertices = np.array(
        [
            # Bottom-left block
            [0, 0, 0],
            [2, 0, 0],
            [2, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 1],
            [2, 1, 1],
            [0, 1, 1],
            # Top-left block (extends upward)
            [0, 1, 0],
            [1, 1, 0],
            [1, 2, 0],
            [0, 2, 0],
            [0, 1, 1],
            [1, 1, 1],
            [1, 2, 1],
            [0, 2, 1],
        ],
        dtype=np.float32,
    )
    # Merge shared vertices (indices 3=8, 7=12)
    # Keep as separate blocks with shared-vertex faces for clarity
    faces = np.array(
        [
            # Bottom block faces
            [0, 1, 2],
            [0, 2, 3],
            [4, 7, 6],
            [4, 6, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 6, 7],
            [2, 7, 3],
            [0, 3, 7],
            [0, 7, 4],
            [1, 5, 6],
            [1, 6, 2],
            # Top block faces
            [8, 9, 10],
            [8, 10, 11],
            [12, 15, 14],
            [12, 14, 13],
            [8, 12, 13],
            [8, 13, 9],
            [10, 14, 15],
            [10, 15, 11],
            [8, 11, 15],
            [8, 15, 12],
            [9, 13, 14],
            [9, 14, 10],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def thin_panel_mesh(
    width: float = 2.0,
    height: float = 2.0,
    thickness: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Thin panel — tests near-degenerate geometry handling."""
    hw, hh, ht = width / 2, height / 2, thickness / 2
    vertices = np.array(
        [
            [-hw, -hh, -ht],
            [hw, -hh, -ht],
            [hw, hh, -ht],
            [-hw, hh, -ht],
            [-hw, -hh, ht],
            [hw, -hh, ht],
            [hw, hh, ht],
            [-hw, hh, ht],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 7, 6],
            [4, 6, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 6, 7],
            [2, 7, 3],
            [0, 3, 7],
            [0, 7, 4],
            [1, 5, 6],
            [1, 6, 2],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def disconnected_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Two separated boxes. Tests component splitting."""
    v1, f1 = box_mesh((1.0, 1.0, 1.0))
    v2, f2 = box_mesh((1.0, 1.0, 1.0))
    v2 = v2 + np.array([5.0, 0.0, 0.0], dtype=np.float32)
    vertices = np.concatenate([v1, v2])
    faces = np.concatenate([f1, f2 + len(v1)])
    return vertices, faces


def degenerate_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Box with one degenerate (zero-area) face appended."""
    v, f = box_mesh()
    degen_face = np.array([[0, 0, 1]], dtype=np.int32)
    faces = np.concatenate([f, degen_face])
    return v, faces


def high_complexity_mesh(subdivisions: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Icosphere with many faces — stress test for decomposition.

    Pure numpy icosphere (no trimesh dependency for fixture generation).
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Icosahedron base vertices
    verts = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    # Normalize to unit sphere
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    tris = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )

    # Subdivide
    for _ in range(subdivisions):
        edge_midpoints: dict[tuple[int, int], int] = {}
        new_tris = []
        verts_list = list(verts)

        for tri in tris:
            mids = []
            for j in range(3):
                a, b = int(tri[j]), int(tri[(j + 1) % 3])
                key = (min(a, b), max(a, b))
                if key not in edge_midpoints:
                    mid = (np.array(verts_list[a]) + np.array(verts_list[b])) / 2
                    mid /= np.linalg.norm(mid)
                    edge_midpoints[key] = len(verts_list)
                    verts_list.append(mid)
                mids.append(edge_midpoints[key])

            v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])
            m01, m12, m20 = mids
            new_tris.extend(
                [
                    [v0, m01, m20],
                    [v1, m12, m01],
                    [v2, m20, m12],
                    [m01, m12, m20],
                ]
            )

        verts = np.array(verts_list, dtype=np.float64)
        tris = np.array(new_tris, dtype=np.int32)

    return verts.astype(np.float32), tris


def _ear_clip(poly: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation for a simple (possibly concave) 2D polygon.

    `poly` must be wound CCW. Returns index triples into `poly`.
    """
    n = len(poly)
    indices = list(range(n))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def is_convex(i0, i1, i2):
        return cross(poly[i0], poly[i1], poly[i2]) > 1e-12

    def point_in_tri(p, a, b, c):
        d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
        has_neg = d1 < 0 or d2 < 0 or d3 < 0
        has_pos = d1 > 0 or d2 > 0 or d3 > 0
        return not (has_neg and has_pos)

    triangles: list[tuple[int, int, int]] = []
    while len(indices) > 3:
        m = len(indices)
        ear_found = False
        for k in range(m):
            i0, i1, i2 = indices[(k - 1) % m], indices[k], indices[(k + 1) % m]
            if not is_convex(i0, i1, i2):
                continue
            a, b, c = poly[i0], poly[i1], poly[i2]
            ear = True
            for idx in indices:
                if idx in (i0, i1, i2):
                    continue
                if point_in_tri(poly[idx], a, b, c):
                    ear = False
                    break
            if ear:
                triangles.append((i0, i1, i2))
                del indices[k]
                ear_found = True
                break
        if not ear_found:
            # Degenerate/numerically awkward remainder: fan-triangulate as a fallback.
            for k in range(1, m - 1):
                triangles.append((indices[0], indices[k], indices[k + 1]))
            indices = []
            break
    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))
    return triangles


def _extrude_polygon(
    poly2d: list[tuple[float, float]], z0: float, z1: float
) -> tuple[np.ndarray, np.ndarray]:
    """Extrude a CCW-wound simple 2D polygon along z into a watertight solid."""
    n = len(poly2d)
    bottom = [(x, y, z0) for x, y in poly2d]
    top = [(x, y, z1) for x, y in poly2d]
    vertices = np.array(bottom + top, dtype=np.float32)

    faces = []
    for i0, i1, i2 in _ear_clip(poly2d):
        faces.append((i0 + n, i1 + n, i2 + n))  # top cap, outward +z
        faces.append((i0, i2, i1))  # bottom cap, outward -z
    for j in range(n):
        j1 = (j + 1) % n
        b0, b1, t0, t1 = j, j1, j + n, j1 + n
        faces.append((b0, b1, t1))
        faces.append((b0, t1, t0))

    return vertices, np.array(faces, dtype=np.int32)


def thin_u_channel_mesh(
    width: float = 2.0,
    height: float = 1.0,
    depth: float = 1.0,
    wall_thickness: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Open-top U-channel/gutter — nonconvex, thin-walled, extruded along depth."""
    hw = width / 2
    t = wall_thickness
    profile = [
        (-hw, 0.0),
        (hw, 0.0),
        (hw, height),
        (hw - t, height),
        (hw - t, t),
        (-hw + t, t),
        (-hw + t, height),
        (-hw, height),
    ]
    return _extrude_polygon(profile, 0.0, depth)


def curved_pipe_quarter_mesh(
    inner_radius: float = 0.5,
    outer_radius: float = 0.7,
    thickness: float = 0.2,
    segments: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Quarter-torus pipe-bend wedge — rectangular cross-section swept 90 degrees."""
    half_t = thickness / 2
    angles = np.linspace(0.0, np.pi / 2, segments + 1)

    verts = []
    for a in angles:
        c, s = np.cos(a), np.sin(a)
        verts.append([inner_radius * c, inner_radius * s, -half_t])  # 0: inner-bottom
        verts.append([outer_radius * c, outer_radius * s, -half_t])  # 1: outer-bottom
        verts.append([outer_radius * c, outer_radius * s, half_t])  # 2: outer-top
        verts.append([inner_radius * c, inner_radius * s, half_t])  # 3: inner-top
    vertices = np.array(verts, dtype=np.float32)

    faces = []
    for i in range(segments):
        r0, r1 = i * 4, (i + 1) * 4
        ib0, ob0, ot0, it0 = r0, r0 + 1, r0 + 2, r0 + 3
        ib1, ob1, ot1, it1 = r1, r1 + 1, r1 + 2, r1 + 3
        # Outer wall (normal points away from bend axis).
        faces += [(ob0, ob1, ot1), (ob0, ot1, ot0)]
        # Inner wall (normal points toward bend axis — concave surface).
        faces += [(ib0, it0, it1), (ib0, it1, ib1)]
        # Bottom wall (z = -half_t, normal -z).
        faces += [(ib0, ob1, ob0), (ib0, ib1, ob1)]
        # Top wall (z = +half_t, normal +z).
        faces += [(it0, ot0, ot1), (it0, ot1, it1)]

    # End caps at theta=0 and theta=90 degrees.
    r0 = 0
    faces += [(r0, r0 + 2, r0 + 1), (r0, r0 + 3, r0 + 2)]
    rlast = segments * 4
    faces += [(rlast, rlast + 1, rlast + 2), (rlast, rlast + 2, rlast + 3)]

    return vertices, np.array(faces, dtype=np.int32)


def cross_bracket_mesh(
    arm_length: float = 1.0,
    arm_width: float = 0.3,
    thickness: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Plus/cross (+) shaped bracket extruded along thickness — 4 concave inner corners."""
    hw = arm_width / 2
    length = arm_length
    profile = [
        (hw, length),
        (-hw, length),
        (-hw, hw),
        (-length, hw),
        (-length, -hw),
        (-hw, -hw),
        (-hw, -length),
        (hw, -length),
        (hw, -hw),
        (length, -hw),
        (length, hw),
        (hw, hw),
    ]
    return _extrude_polygon(profile, 0.0, thickness)


def h_shape_mesh(
    arm_length: float = 1.0,
    arm_width: float = 0.3,
    bar_height: float = 0.4,
    thickness: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """H-shaped extrusion — two vertical bars joined by a horizontal bar.

    Multiple concavities at bar-arm junctions. Guaranteed watertight via
    ``_extrude_polygon``.
    """
    hw = arm_width / 2
    bh = bar_height / 2
    profile = [
        (-arm_length / 2, -hw),
        (-arm_length / 2, hw),
        (-bh, hw),
        (-bh, arm_length / 2),
        (bh, arm_length / 2),
        (bh, hw),
        (arm_length / 2, hw),
        (arm_length / 2, -hw),
        (bh, -hw),
        (bh, -arm_length / 2),
        (-bh, -arm_length / 2),
        (-bh, -hw),
    ]
    return _extrude_polygon(profile, 0.0, thickness)


def oblique_gear_prism_mesh() -> tuple[np.ndarray, np.ndarray]:
    """13-tooth asymmetric gear prism — many concave corners, anisotropic transform."""
    n_teeth = 13
    base_r = 0.8
    tip_r_even = 1.2
    tip_r_odd = 1.05
    half_width = 0.06  # half of the 0.12 rad tooth angular width

    centers = [i * 2 * np.pi / n_teeth + 0.05 * np.sin(i * 1.7) for i in range(n_teeth)]

    profile: list[tuple[float, float]] = []
    for i in range(n_teeth):
        c = centers[i]
        tip_r = tip_r_even if i % 2 == 0 else tip_r_odd
        bl_angle = c - half_width
        tl_angle = c - half_width / 2
        tr_angle = c + half_width / 2
        br_angle = c + half_width

        profile.append((base_r * np.cos(bl_angle), base_r * np.sin(bl_angle)))
        profile.append((tip_r * np.cos(tl_angle), tip_r * np.sin(tl_angle)))
        profile.append((tip_r * np.cos(tr_angle), tip_r * np.sin(tr_angle)))
        profile.append((base_r * np.cos(br_angle), base_r * np.sin(br_angle)))

        next_c = centers[(i + 1) % n_teeth]
        if i == n_teeth - 1:
            next_c += 2 * np.pi
        mid_angle = (c + next_c) / 2
        profile.append((base_r * np.cos(mid_angle), base_r * np.sin(mid_angle)))

    vertices, faces = _extrude_polygon(profile, 0.0, 0.5)

    scale = np.diag([1.0, 0.7, 1.3]).astype(np.float32)
    angle = np.radians(15)
    rot_z = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    vertices = (rot_z @ scale @ vertices.T).T.astype(np.float32)
    return vertices, faces


def twisted_notched_column_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Notched rectangle swept with 60° twist and 25% taper — oblique nonparallel faces."""
    hw, hh = 0.5, 0.4
    nw, nd = 0.15, 0.2
    profile_base = [
        (hw, -hh),
        (hw, hh),
        (nw, hh),  # right edge of top notch
        (nw, hh - nd),  # notch floor
        (-nw, hh - nd),
        (-nw, hh),  # left edge of top notch
        (-hw, hh),
        (-hw, -hh),
    ]
    n_profile = len(profile_base)

    height = 2.0
    stations = 24
    total_twist = np.pi / 3
    taper = 0.25

    station_pts: list[list[tuple[float, float, float]]] = []
    for k in range(stations + 1):
        z = k * height / stations
        twist = total_twist * z / height
        scale_factor = 1.0 - taper * z / height
        c, s = np.cos(twist), np.sin(twist)
        pts = []
        for x, y in profile_base:
            xr = x * c - y * s
            yr = x * s + y * c
            pts.append((scale_factor * xr, scale_factor * yr, z))
        station_pts.append(pts)

    vertices = np.array(
        [pt for station in station_pts for pt in station], dtype=np.float32
    )

    faces = []
    for k in range(stations):
        base_k = k * n_profile
        base_k1 = (k + 1) * n_profile
        for j in range(n_profile):
            j1 = (j + 1) % n_profile
            b0, b1 = base_k + j, base_k + j1
            t0, t1 = base_k1 + j, base_k1 + j1
            faces.append((b0, b1, t1))
            faces.append((b0, t1, t0))

    # Bottom cap (station 0), outward -Z normal.
    for i0, i1, i2 in _ear_clip(profile_base):
        faces.append((i0, i2, i1))

    # Top cap (last station), outward +Z normal.
    top_offset = stations * n_profile
    top_profile_2d = [(p[0], p[1]) for p in station_pts[stations]]
    for i0, i1, i2 in _ear_clip(top_profile_2d):
        faces.append((i0 + top_offset, i1 + top_offset, i2 + top_offset))

    return vertices, np.array(faces, dtype=np.int32)


def skewed_rectangular_torus_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Rectangular-section torus (genus-one) with skew — exercises boundary topology."""
    inner_radius = 0.4
    outer_radius = 0.6
    thickness = 0.2
    segments = 32
    half_t = thickness / 2

    angles = [i * 2 * np.pi / segments for i in range(segments)]

    verts = []
    for a in angles:
        c, s = np.cos(a), np.sin(a)
        verts.append([inner_radius * c, inner_radius * s, -half_t])  # inner-bottom
        verts.append([outer_radius * c, outer_radius * s, -half_t])  # outer-bottom
        verts.append([outer_radius * c, outer_radius * s, half_t])  # outer-top
        verts.append([inner_radius * c, inner_radius * s, half_t])  # inner-top
    vertices = np.array(verts, dtype=np.float32)

    faces = []
    for i in range(segments):
        i1 = (i + 1) % segments
        r0, r1 = i * 4, i1 * 4
        ib0, ob0, ot0, it0 = r0, r0 + 1, r0 + 2, r0 + 3
        ib1, ob1, ot1, it1 = r1, r1 + 1, r1 + 2, r1 + 3
        # Outer wall.
        faces += [(ob0, ob1, ot1), (ob0, ot1, ot0)]
        # Inner wall.
        faces += [(ib0, it0, it1), (ib0, it1, ib1)]
        # Bottom wall.
        faces += [(ib0, ob1, ob0), (ib0, ib1, ob1)]
        # Top wall.
        faces += [(it0, ot0, ot1), (it0, ot1, it1)]

    vertices[:, 0] += 0.15 * vertices[:, 2]

    return vertices, np.array(faces, dtype=np.int32)


def _orient_convex_faces_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Flip any triangle of a convex point set whose winding faces the centroid."""
    centroid = vertices.mean(axis=0)
    faces = faces.copy()
    for idx in range(len(faces)):
        i0, i1, i2 = faces[idx]
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
        normal = np.cross(v1 - v0, v2 - v0)
        face_centroid = (v0 + v1 + v2) / 3.0
        if np.dot(normal, face_centroid - centroid) < 0:
            faces[idx] = [i0, i2, i1]
    return faces


def multiscale_shard_cluster_mesh() -> tuple[np.ndarray, np.ndarray]:
    """5 irregular polyhedra at 1:32 scale range — component and scale interactions."""

    def rot_x(deg: float) -> np.ndarray:
        a = np.radians(deg)
        return np.array(
            [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]],
            dtype=np.float32,
        )

    def rot_y(deg: float) -> np.ndarray:
        a = np.radians(deg)
        return np.array(
            [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]],
            dtype=np.float32,
        )

    def rot_z(deg: float) -> np.ndarray:
        a = np.radians(deg)
        return np.array(
            [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]],
            dtype=np.float32,
        )

    # Shard 1: irregular tetrahedron.
    v1 = np.array(
        [[0, 0, 0], [1.2, 0.1, 0.1], [0.3, 1.1, 0.2], [0.1, 0.2, 0.9]],
        dtype=np.float32,
    )
    f1 = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)

    # Shard 2: triangular prism/wedge.
    v2 = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0.3, 0.9, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0.3, 0.9, 1],
        ],
        dtype=np.float32,
    )
    f2 = np.array(
        [
            [0, 1, 2],
            [3, 5, 4],
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [2, 0, 3],
            [2, 3, 5],
        ],
        dtype=np.int32,
    )

    # Shard 3: skewed box (one corner displaced).
    hx, hy, hz = 0.5, 0.5, 0.5
    v3 = np.array(
        [
            [-hx, -hy, -hz],
            [-hx, -hy, hz],
            [-hx, hy, -hz],
            [-hx, hy, hz],
            [hx, -hy, -hz],
            [hx, -hy, hz],
            [hx, hy, -hz],
            [0.9, 0.9, 0.9],  # displaced corner
        ],
        dtype=np.float32,
    )
    f3 = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        dtype=np.int32,
    )

    # Shard 4: elongated tetrahedron.
    v4 = np.array(
        [[0, 0, 0], [3.0, 0.05, 0.05], [0.2, 0.3, 0.1], [0.15, 0.1, 0.4]],
        dtype=np.float32,
    )
    f4 = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)

    # Shard 5: flat wedge.
    v5 = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 0.8, 0],
            [0, 0, 0.05],
            [1, 0, 0.05],
            [0.5, 0.8, 0.05],
        ],
        dtype=np.float32,
    )
    f5 = np.array(
        [
            [0, 1, 2],
            [3, 5, 4],
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [2, 0, 3],
            [2, 3, 5],
        ],
        dtype=np.int32,
    )

    shards = [
        (v1, f1, 1.0, np.eye(3, dtype=np.float32), (0.0, 0.0, 0.0)),
        (v2, f2, 0.5, rot_z(90.0), (5.0, 0.0, 0.0)),
        (v3, f3, 0.25, rot_x(90.0), (0.0, 5.0, 0.0)),
        (v4, f4, 0.125, rot_y(45.0), (0.0, 0.0, 5.0)),
        (v5, f5, 0.03125, rot_y(90.0), (5.0, 5.0, 5.0)),
    ]

    all_vertices = []
    all_faces = []
    vertex_offset = 0
    for v, f, scale, rot, translate in shards:
        t = np.array(translate, dtype=np.float32)
        v_t = ((rot @ (v * scale).T).T + t).astype(np.float32)
        f_fixed = _orient_convex_faces_outward(v_t, f)
        all_vertices.append(v_t)
        all_faces.append(f_fixed + vertex_offset)
        vertex_offset += len(v_t)

    vertices = np.concatenate(all_vertices).astype(np.float32)
    faces = np.concatenate(all_faces).astype(np.int32)
    return vertices, faces


# Registry of all fixtures
FIXTURES = {
    "box": box_mesh,
    "l_shape": l_shape_mesh,
    "thin_panel": thin_panel_mesh,
    "disconnected": disconnected_mesh,
    "degenerate": degenerate_mesh,
    "high_complexity": high_complexity_mesh,
    "thin_u_channel": thin_u_channel_mesh,
    "curved_pipe_quarter": curved_pipe_quarter_mesh,
    "cross_bracket": cross_bracket_mesh,
    "h_shape": h_shape_mesh,
}

HOLDOUT_FIXTURES = {
    "oblique_gear_prism": oblique_gear_prism_mesh,
    "twisted_notched_column": twisted_notched_column_mesh,
    "skewed_rectangular_torus": skewed_rectangular_torus_mesh,
    "multiscale_shard_cluster": multiscale_shard_cluster_mesh,
}
