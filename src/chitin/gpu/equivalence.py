"""Oracle-vs-GPU equivalence contract for the GPU clipping pipeline gate.

This module defines what "the GPU result agrees with the CPU f32 oracle"
means, and it is what the gate verdict (`scripts/evaluate_gpu_gate.py`,
`docs/gpu-clipping-pipeline-contract.md`) depends on:

* the plane corpus and calibration fixtures the gate evaluates
  (`generate_planes`, `CALIBRATION_FIXTURES`);
* the CPU oracle run on each (mesh, plane) pair (`run_cpu_oracle`);
* canonical vertex identities (`canonicalize_vertex`): source vertices by
  original index, intersection vertices by EdgeKey, never by position;
* topology parity, per-loop cap equivalence, cap winding, and the per-clip
  criteria (`check_topology_parity`, `compare_caps_equivalent`,
  `check_side_loop_equivalence`, `check_cap_winding`, `check_criteria`).

Changing any of these changes what counts as agreement, or which clips are
evaluated, and therefore re-bases every banked calibration result
(`docs/gpu-gate-calibration-results.json`). Do not edit it to make a failing
clip pass; a change here needs a new calibration run and a new banked record.
The `evaluator_digest` in each results file pins this file together with
`scripts/evaluate_gpu_gate.py` (`evaluator_files` lists them in digest order),
so a banked verdict names the exact contract that produced it.

Stage-level divergence attribution (`chitin.gpu.clip_diagnostics`) is
diagnostic only and does not affect the verdict, but it shares
`canonicalize_vertex` so both views use one identity definition.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from fractions import Fraction

import numpy as np

from chitin.f32_policy import POLICY_0_3_0
from chitin.f32_predicates import (
    classify_plane_f32,
    clip_mesh_f32,
    extract_cap_f32,
)
from chitin.gpu.pipeline import ClipPipelineResult, ClipStatus
from chitin.trace_fixtures import (
    box_mesh,
    cross_bracket_mesh,
    curved_pipe_quarter_mesh,
    h_shape_calibration_mesh,
)

__all__ = [
    "CALIBRATION_FIXTURES",
    "canonicalize_vertex",
    "check_cap_winding",
    "check_criteria",
    "check_side_loop_equivalence",
    "check_topology_parity",
    "compare_caps_equivalent",
    "generate_planes",
    "run_cpu_oracle",
]

# NOTE: the gate contract (docs/gpu-clipping-pipeline-contract.md) names the
# calibration corpus "t_shape, curved_pipe, h_shape" -- shorthand from the
# Policy 0.1/0.2 gates. `chitin.trace_fixtures` has never exported a
# `t_shape_mesh`; the closest surviving fixture from that original corpus
# (`FIXTURES` dict) with an analogous concave junction is `cross_bracket_mesh`,
# used here in its place and reported under its real name so the results
# file never claims a fixture that wasn't actually run.
CALIBRATION_FIXTURES = {
    "box": box_mesh,
    "cross_bracket": cross_bracket_mesh,
    "curved_pipe_quarter": curved_pipe_quarter_mesh,
    "h_shape_corrected": h_shape_calibration_mesh,
}


def generate_planes(
    vertices: np.ndarray, fixture_name: str, count: int = 200
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate `count` deterministic clipping planes for a fixture.

    Four categories, each `count // 4` planes (remainder goes to the last
    category): random planes through the bbox center, planes through random
    mesh vertices (on-plane stress), axis-aligned planes at random heights,
    and random planes through random bbox positions.
    """
    seed = int.from_bytes(hashlib.sha256(fixture_name.encode()).digest()[:8], "little")
    rng = np.random.Generator(np.random.PCG64(seed))

    v64 = vertices.astype(np.float64)
    bbox_min = v64.min(axis=0)
    bbox_max = v64.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    n_verts = len(v64)

    def random_unit_normal() -> np.ndarray:
        while True:
            v = rng.normal(size=3)
            norm = np.linalg.norm(v)
            if norm > 1e-9:
                return v / norm

    n_each = count // 4
    n_remaining = count - 3 * n_each

    raw_planes: list[tuple[np.ndarray, np.ndarray]] = []

    # 1. Random planes through the mesh bounding-box center.
    for _ in range(n_each):
        raw_planes.append((center.copy(), random_unit_normal()))

    # 2. Planes through random mesh vertices (on-plane stress testing).
    for _ in range(n_each):
        idx = rng.integers(0, n_verts)
        raw_planes.append((v64[idx].copy(), random_unit_normal()))

    # 3. Axis-aligned planes at random heights within the bbox.
    for _ in range(n_each):
        axis = int(rng.integers(0, 3))
        normal = np.zeros(3)
        normal[axis] = 1.0
        point = center.copy()
        point[axis] = rng.uniform(bbox_min[axis], bbox_max[axis])
        raw_planes.append((point, normal))

    # 4. Remaining: random planes through random bbox positions.
    for _ in range(n_remaining):
        point = rng.uniform(bbox_min, bbox_max)
        raw_planes.append((point, random_unit_normal()))

    planes: list[tuple[np.ndarray, np.ndarray]] = []
    for point, normal in raw_planes:
        n64 = normal.astype(np.float64)
        n64 = n64 / np.linalg.norm(n64)
        planes.append((point.astype(np.float32), n64.astype(np.float32)))
    return planes


def _orient_oracle_caps(cap, vertices, normal):
    """Orient whole loop triangulations; never repair individual flipped faces."""
    faces = cap.cap_faces.copy()
    for loop in cap.loops:
        mask = np.all(np.isin(faces, loop), axis=1)
        triangles = vertices[faces[mask]]
        if len(triangles) == 0:
            continue
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        if np.dot(normals.sum(axis=0), normal) < 0:
            faces[mask] = faces[mask][:, [0, 2, 1]]
    return faces


def run_cpu_oracle(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> dict:
    """Run the CPU f32 oracle on a single (mesh, plane) pair, both halves."""
    policy = POLICY_0_3_0
    # Positive half (sign >= 0).
    pos_clip = clip_mesh_f32(vertices, faces, plane_point, plane_normal, policy)
    pos_cap = extract_cap_f32(pos_clip, policy)
    # Negative half (sign <= 0) -- negate the normal so clip_mesh_f32's
    # "positive" side becomes the mesh's negative side.
    neg_clip = clip_mesh_f32(vertices, faces, plane_point, -plane_normal, policy)
    neg_cap = extract_cap_f32(neg_clip, policy)
    classification = classify_plane_f32(vertices, plane_point, plane_normal, policy)

    from chitin.cap_triangulation import triangulate_polygon
    from chitin.surface_boundary import surface_loops

    for clip, cap in ((pos_clip, pos_cap), (neg_clip, neg_cap)):
        cap.loops = surface_loops(clip.faces)
        axes = [i for i in range(3) if i != int(np.argmax(np.abs(plane_normal)))]
        cap.cap_faces = np.asarray(
            [
                face
                for loop in cap.loops
                for face in triangulate_polygon(loop, clip.vertices[loop][:, axes])
            ],
            dtype=np.int64,
        ).reshape(-1, 3)

    return {
        "signs": classification.signs,
        "pos_vertices": pos_clip.vertices,
        "pos_faces": pos_clip.faces,
        "pos_cap_faces": _orient_oracle_caps(pos_cap, pos_clip.vertices, plane_normal),
        "pos_loops": pos_cap.loops,
        "neg_vertices": neg_clip.vertices,
        "neg_faces": neg_clip.faces,
        "neg_cap_faces": _orient_oracle_caps(neg_cap, neg_clip.vertices, -plane_normal),
        "neg_loops": neg_cap.loops,
        "pos_edge_key_map": pos_clip.edge_key_map,
        "neg_edge_key_map": neg_clip.edge_key_map,
        "pos_ancestry": pos_clip.ancestry,
        "neg_ancestry": neg_clip.ancestry,
    }


def canonicalize_vertex(v: int, vertex_count: int, edge_keys: np.ndarray) -> tuple:
    """Convert a vertex index to its canonical identity.

    Source vertex (v < vertex_count): (0, v).
    Intersection vertex: (1, v_lo, v_hi), the EdgeKey it lies on.
    An intersection index past the edge-key map is unmapped: (-1, v).
    """
    if v < vertex_count:
        return (0, int(v))
    ek_idx = v - vertex_count
    if ek_idx >= len(edge_keys):
        return (-1, int(v))
    return (1, int(edge_keys[ek_idx, 0]), int(edge_keys[ek_idx, 1]))


def _face_winding_sign(
    face: np.ndarray, vertices: np.ndarray, plane_normal: np.ndarray
) -> int:
    """+1 if face normal aligns with plane_normal, -1 if opposed, 0 if indeterminate.

    Returns 0 for degenerate faces (zero-area) and for sliver triangles
    whose normal is nearly perpendicular to the plane normal (|dot| < 0.01).
    Clipped slivers tangent to the plane have f32-unstable winding signs
    that differ between GPU and CPU implementations of the same lerp.
    """
    tri = vertices[face].astype(np.float64)
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    nl = np.linalg.norm(normal)
    if nl < 1e-12:
        return 0
    dot = np.dot(normal / nl, plane_normal.astype(np.float64))
    if abs(dot) < 0.01:
        return 0
    return 1 if dot > 0 else -1


def _canonical_face_key(
    face: np.ndarray,
    ancestry_val: int,
    vertices: np.ndarray,
    vertex_count: int,
    edge_keys: np.ndarray,
    plane_normal: np.ndarray,
) -> tuple:
    """Canonical face identity: (sorted_vertex_ids, winding, source_face)."""
    cids = tuple(
        sorted(canonicalize_vertex(int(v), vertex_count, edge_keys) for v in face)
    )
    winding = _face_winding_sign(face, vertices, plane_normal)
    return (cids, winding, int(ancestry_val))


def _compare_side_canonical(
    gpu_verts: np.ndarray,
    gpu_faces: np.ndarray,
    gpu_ancestry: np.ndarray,
    gpu_vertex_count: int,
    gpu_edge_keys: np.ndarray,
    cpu_verts: np.ndarray,
    cpu_faces: np.ndarray,
    cpu_ancestry: np.ndarray,
    cpu_vertex_count: int,
    cpu_edge_keys: np.ndarray,
    plane_normal: np.ndarray,
) -> tuple[int, int]:
    """Canonical-identity multiset comparison of clip faces."""
    if len(gpu_faces) == 0 and len(cpu_faces) == 0:
        return 0, 0

    gpu_counter: Counter[tuple] = Counter()
    for i, f in enumerate(gpu_faces):
        key = _canonical_face_key(
            f,
            int(gpu_ancestry[i]),
            gpu_verts,
            gpu_vertex_count,
            gpu_edge_keys,
            plane_normal,
        )
        gpu_counter[key] += 1

    cpu_counter: Counter[tuple] = Counter()
    for i, f in enumerate(cpu_faces):
        key = _canonical_face_key(
            f,
            int(cpu_ancestry[i]),
            cpu_verts,
            cpu_vertex_count,
            cpu_edge_keys,
            plane_normal,
        )
        cpu_counter[key] += 1

    if gpu_counter == cpu_counter:
        return sum(gpu_counter.values()), 0

    common = gpu_counter & cpu_counter
    agree = sum(common.values())
    disagree = sum((gpu_counter - cpu_counter).values()) + sum(
        (cpu_counter - gpu_counter).values()
    )
    return agree, disagree


def _compare_caps_strict(
    gpu_verts: np.ndarray,
    gpu_cap_faces: np.ndarray,
    gpu_vertex_count: int,
    gpu_edge_keys: np.ndarray,
    cpu_verts: np.ndarray,
    cpu_cap_faces: np.ndarray,
    cpu_vertex_count: int,
    cpu_edge_keys: np.ndarray,
    plane_normal: np.ndarray,
) -> tuple[int, int]:
    """Strict canonical-identity comparison of cap faces (informational only).

    Cap triangulation is not unique -- a convex or concave boundary loop
    admits many valid ear-clip fans -- so this strict face-set parity is no
    longer the gate criterion (see `compare_caps_equivalent`). Retained to
    report a strict-match metric alongside the equivalence verdict.

    Cap ancestry is 0xFFFFFFFF on both sides.
    """
    if len(gpu_cap_faces) == 0 and len(cpu_cap_faces) == 0:
        return 0, 0

    cap_ancestry = 0xFFFFFFFF

    gpu_counter: Counter[tuple] = Counter()
    for f in gpu_cap_faces:
        key = _canonical_face_key(
            f,
            cap_ancestry,
            gpu_verts,
            gpu_vertex_count,
            gpu_edge_keys,
            plane_normal,
        )
        gpu_counter[key] += 1

    cpu_counter: Counter[tuple] = Counter()
    for f in cpu_cap_faces:
        key = _canonical_face_key(
            f,
            cap_ancestry,
            cpu_verts,
            cpu_vertex_count,
            cpu_edge_keys,
            plane_normal,
        )
        cpu_counter[key] += 1

    if gpu_counter == cpu_counter:
        return sum(gpu_counter.values()), 0

    common = gpu_counter & cpu_counter
    agree = sum(common.values())
    disagree = sum((gpu_counter - cpu_counter).values()) + sum(
        (cpu_counter - gpu_counter).values()
    )
    return agree, disagree


def _rotate_loop_to_smallest(seq: tuple) -> tuple:
    """Canonicalize an undirected cycle over rotations and reversal.

    Loop traversal direction is arbitrary; cap winding is checked separately.
    """
    if not seq:
        return seq
    forward = min(seq[i:] + seq[:i] for i in range(len(seq)))
    reverse = seq[::-1]
    backward = min(reverse[i:] + reverse[:i] for i in range(len(reverse)))
    return min(forward, backward)


def _loop_boundary_edge_set(loop: np.ndarray) -> set[frozenset]:
    n = len(loop)
    return {frozenset((int(loop[i]), int(loop[(i + 1) % n]))) for i in range(n)}


def _segment_intersect_2d(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> bool:
    """Proper intersection test for segments ab and cd in 2D (f64).

    Shared endpoints do not count as a crossing -- only a transversal
    intersection of the open segments does.
    """
    if np.array_equal(a, c) or np.array_equal(a, d):
        return False
    if np.array_equal(b, c) or np.array_equal(b, d):
        return False

    def cross(o: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    d1 = cross(c, d, a)
    d2 = cross(c, d, b)
    d3 = cross(a, b, c)
    d4 = cross(a, b, d)
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    )


def _point_in_polygon_2d(pt: np.ndarray, poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test. `poly` is (n, 2) f64, implicitly closed."""
    x, y = float(pt[0]), float(pt[1])
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _polygon_area_2d(poly2d: np.ndarray) -> float:
    x = poly2d[:, 0]
    y = poly2d[:, 1]
    return abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))) / 2.0


def _triangle_area_2d(tri2d: np.ndarray) -> float:
    a, b, c = tri2d[0], tri2d[1], tri2d[2]
    return (
        abs(float((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))) / 2.0
    )


def check_side_loop_equivalence(
    loop: np.ndarray,
    cap_faces: np.ndarray,
    verts: np.ndarray,
    plane_normal: np.ndarray,
    expected_winding_sign: int = 0,
) -> tuple[bool, dict]:
    """Check that `cap_faces` triangulate `loop` as a valid, equivalent cap.

    Any valid ear-clip/fan triangulation of the same boundary loop passes:
    correct triangle count, nondegenerate triangles, each boundary edge
    used once and each internal diagonal used twice with opposite
    orientation, no crossing diagonals, no diagonal crossing a boundary
    edge, consistent winding matching the expected side, every triangle
    centroid contained in the polygon (2D projection), and matching area.

    `expected_winding_sign`: +1 for positive-half caps (normals should
    align with plane_normal), -1 for negative-half caps, 0 for
    self-consistency-only (no polarity enforcement).
    """
    detail: dict = {}
    n = len(loop)
    loop_set = {int(v) for v in loop}
    verts64 = verts.astype(np.float64)

    tris = [f for f in cap_faces if all(int(v) in loop_set for v in f)]
    detail["triangle_count"] = len(tris)
    detail["expected_triangle_count"] = n - 2

    # Condition 2: exactly n-2 triangles for an n-vertex loop.
    cond2 = len(tris) == (n - 2)

    # Scale-relative degeneracy threshold: cross-product magnitude of
    # two f32 edges at the loop's bounding-box scale.  d² * 1e-5 is a
    # generous noise floor for the cross product of two edges, each of
    # length at most d, under f32 rounding.
    loop_verts64 = verts64[np.asarray(loop, dtype=np.int64)]
    bbox_diag = float(
        np.linalg.norm(loop_verts64.max(axis=0) - loop_verts64.min(axis=0))
    )
    degen_threshold = bbox_diag * bbox_diag * 1e-5

    # Condition 3: exact nonzero area of the supplied binary coordinates.
    # The global loop-scale floor measures mesh quality, not degeneracy.
    areas: list[float] = []
    normals: list[np.ndarray] = []
    cond3 = True
    low_margin_count = 0
    for f in tris:
        tri = verts64[np.asarray(f, dtype=np.int64)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        mag = float(np.linalg.norm(normal))
        normals.append(normal)
        areas.append(0.5 * mag)
        if mag <= degen_threshold:
            low_margin_count += 1
            a, b, c = [[Fraction(float(x)) for x in point] for point in tri]
            u = [b[i] - a[i] for i in range(3)]
            v = [c[i] - a[i] for i in range(3)]
            if all(u[i] * v[(i + 1) % 3] == u[(i + 1) % 3] * v[i] for i in range(3)):
                cond3 = False

    # Conditions 4-5: boundary edges used exactly once, internal diagonals
    # used exactly twice with opposite orientation.
    boundary_edges = _loop_boundary_edge_set(loop)
    undirected_count: Counter = Counter()
    directed_count: Counter = Counter()
    for f in tris:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            undirected_count[frozenset((a, b))] += 1
            directed_count[(a, b)] += 1

    cond4 = all(undirected_count.get(e, 0) == 1 for e in boundary_edges)
    internal_edges = [e for e in undirected_count if e not in boundary_edges]
    cond5 = all(undirected_count[e] == 2 for e in internal_edges)
    for e in internal_edges:
        a, b = tuple(e)
        if not (
            directed_count.get((a, b), 0) == 1 and directed_count.get((b, a), 0) == 1
        ):
            cond5 = False
            break

    # 2D projection: drop the axis(es) most aligned with plane_normal, i.e.
    # keep the two axes where |plane_normal[axis]| is smallest.
    axes = tuple(
        int(a)
        for a in np.argsort(np.abs(np.asarray(plane_normal, dtype=np.float64)))[:2]
    )
    loop_idx = np.asarray(loop, dtype=np.int64)
    loop2d = verts64[loop_idx][:, axes]
    idx_pos = {int(v): i for i, v in enumerate(loop)}

    # Condition 6: no crossing diagonals.
    diagonals = [tuple(sorted(e)) for e in internal_edges]
    cond6 = True
    for i in range(len(diagonals)):
        a1, b1 = diagonals[i]
        p1, p2 = loop2d[idx_pos[a1]], loop2d[idx_pos[b1]]
        for j in range(i + 1, len(diagonals)):
            a2, b2 = diagonals[j]
            if len({a1, b1, a2, b2}) < 4:
                continue
            p3, p4 = loop2d[idx_pos[a2]], loop2d[idx_pos[b2]]
            if _segment_intersect_2d(p1, p2, p3, p4):
                cond6 = False
                break
        if not cond6:
            break

    # Condition 7: no boundary edge crossed by an internal diagonal.
    boundary_list = [tuple(sorted(e)) for e in boundary_edges]
    cond7 = True
    for a1, b1 in diagonals:
        p1, p2 = loop2d[idx_pos[a1]], loop2d[idx_pos[b1]]
        for a2, b2 in boundary_list:
            if len({a1, b1, a2, b2}) < 4:
                continue
            p3, p4 = loop2d[idx_pos[a2]], loop2d[idx_pos[b2]]
            if _segment_intersect_2d(p1, p2, p3, p4):
                cond7 = False
                break
        if not cond7:
            break

    # Condition 8: consistent winding -- every nondegenerate triangle's
    # normal dots plane_normal with the same sign.  When expected_winding_sign
    # is nonzero, that sign must match the expected side (+1 for positive half,
    # -1 for negative half).  When 0, only self-consistency is required.
    plane_n64 = np.asarray(plane_normal, dtype=np.float64)
    ref_sign = expected_winding_sign
    cond8 = True
    for normal, area in zip(normals, areas):
        dot = float(np.dot(normal, plane_n64))
        sign = 1 if dot > 0 else (-1 if dot < 0 else 0)
        if sign == 0:
            cond8 = False
            break
        if ref_sign == 0:
            ref_sign = sign
        elif sign != ref_sign:
            cond8 = False
            break

    # Condition 9: every triangle centroid projects inside the polygon.
    cond9 = True
    for f in tris:
        tri3d = verts64[np.asarray(f, dtype=np.int64)]
        centroid2d = tri3d[:, axes].mean(axis=0)
        if not _point_in_polygon_2d(centroid2d, loop2d):
            cond9 = False
            break

    # Condition 10: triangle-area sum matches polygon area (both computed
    # in the same 2D projection, so the projection's distortion cancels).
    tri_area_2d_sum = sum(
        _triangle_area_2d(verts64[np.asarray(f, dtype=np.int64)][:, axes]) for f in tris
    )
    poly_area_2d = _polygon_area_2d(loop2d)
    denom = poly_area_2d if poly_area_2d > 1e-12 else 1.0
    cond10 = abs(tri_area_2d_sum - poly_area_2d) / denom < 1e-4

    detail.update(
        {
            "nondegenerate": cond3,
            "low_margin_triangle_count": low_margin_count,
            "boundary_edges_ok": cond4,
            "internal_diagonals_ok": cond5,
            "no_crossing_diagonals": cond6,
            "no_boundary_crossed": cond7,
            "winding_consistent": cond8,
            "centroids_contained": cond9,
            "area_parity": cond10,
            "triangle_area_2d_sum": tri_area_2d_sum,
            "polygon_area_2d": poly_area_2d,
        }
    )
    ok = (
        cond2
        and cond3
        and cond4
        and cond5
        and cond6
        and cond7
        and cond8
        and cond9
        and cond10
    )
    return ok, detail


def compare_caps_equivalent(
    gpu_verts: np.ndarray,
    gpu_cap_faces: np.ndarray,
    gpu_vertex_count: int,
    gpu_edge_keys: np.ndarray,
    cpu_verts: np.ndarray,
    cpu_cap_faces: np.ndarray,
    cpu_vertex_count: int,
    cpu_edge_keys: np.ndarray,
    plane_normal: np.ndarray,
    gpu_loops: list[np.ndarray],
    cpu_loops: list[np.ndarray],
    expected_winding_sign: int = 0,
) -> dict:
    """Cap-triangulation equivalence: same boundary loop(s), any valid
    triangulation of them, rather than strict per-triangle face-set parity.

    GPU and CPU loops are independently generated and may arrive in
    different order. They are matched by canonical cyclic identity —
    each loop is canonicalized across rotations and reversal of canonical
    vertex tuples and then paired across sides. Unmatched loops on
    either side are a failure.
    """
    _, strict_disagree = _compare_caps_strict(
        gpu_verts,
        gpu_cap_faces,
        gpu_vertex_count,
        gpu_edge_keys,
        cpu_verts,
        cpu_cap_faces,
        cpu_vertex_count,
        cpu_edge_keys,
        plane_normal,
    )
    strict_match = strict_disagree == 0

    # Canonicalize each side's loops independently: convert vertex
    # indices to canonical identity tuples and rotate to lex-smallest.
    def _canon_loop(loop: np.ndarray, vc: int, ek: np.ndarray) -> tuple[tuple, ...]:
        return _rotate_loop_to_smallest(
            tuple(canonicalize_vertex(int(v), vc, ek) for v in loop)
        )

    gpu_by_canon: dict[tuple[tuple, ...], np.ndarray] = {}
    for loop in gpu_loops:
        if len(loop) < 3:
            continue
        canon = _canon_loop(loop, gpu_vertex_count, gpu_edge_keys)
        gpu_by_canon[canon] = loop

    cpu_by_canon: dict[tuple[tuple, ...], np.ndarray] = {}
    for loop in cpu_loops:
        if len(loop) < 3:
            continue
        canon = _canon_loop(loop, cpu_vertex_count, cpu_edge_keys)
        cpu_by_canon[canon] = loop

    all_canon_keys = set(gpu_by_canon.keys()) | set(cpu_by_canon.keys())

    details: list[dict] = []
    all_ok = True
    gpu_assigned = 0
    cpu_assigned = 0
    for canon_key in sorted(all_canon_keys):
        loop_detail: dict = {}
        gpu_loop = gpu_by_canon.get(canon_key)
        cpu_loop = cpu_by_canon.get(canon_key)

        if gpu_loop is None or cpu_loop is None:
            loop_detail["error"] = "unmatched_loop"
            loop_detail["gpu_present"] = gpu_loop is not None
            loop_detail["cpu_present"] = cpu_loop is not None
            loop_detail["equivalent"] = False
            details.append(loop_detail)
            all_ok = False
            continue

        if len(gpu_loop) != len(cpu_loop):
            loop_detail["error"] = "loop_length_mismatch"
            loop_detail["gpu_len"] = len(gpu_loop)
            loop_detail["cpu_len"] = len(cpu_loop)
            loop_detail["equivalent"] = False
            details.append(loop_detail)
            all_ok = False
            continue

        loop_detail["loop_identity_match"] = True

        gpu_ok, gpu_detail = check_side_loop_equivalence(
            gpu_loop,
            gpu_cap_faces,
            gpu_verts,
            plane_normal,
            expected_winding_sign,
        )
        cpu_ok, cpu_detail = check_side_loop_equivalence(
            cpu_loop,
            cpu_cap_faces,
            cpu_verts,
            plane_normal,
            expected_winding_sign,
        )
        gpu_assigned += gpu_detail["triangle_count"]
        cpu_assigned += cpu_detail["triangle_count"]
        loop_detail["gpu"] = gpu_detail
        loop_detail["cpu"] = cpu_detail
        loop_ok = gpu_ok and cpu_ok
        loop_detail["equivalent"] = loop_ok
        if not loop_ok:
            all_ok = False
        details.append(loop_detail)

    # Every cap triangle must belong to exactly one loop -- an orphaned or
    # doubly-assigned triangle means the cap does not cover the boundary
    # loops we compared against (condition 1).
    if gpu_assigned != len(gpu_cap_faces) or cpu_assigned != len(cpu_cap_faces):
        all_ok = False
        details.append(
            {
                "error": "uncovered_or_duplicated_cap_triangles",
                "gpu_assigned": gpu_assigned,
                "gpu_total": len(gpu_cap_faces),
                "cpu_assigned": cpu_assigned,
                "cpu_total": len(cpu_cap_faces),
            }
        )

    return {
        "equivalent": all_ok,
        "strict_match": strict_match,
        "loop_count": len(cpu_loops),
        "details": details,
    }


def check_topology_parity(
    gpu_result: ClipPipelineResult,
    cpu_result: dict,
    plane_normal: np.ndarray,
    source_vertex_count: int,
    source_vertices: np.ndarray,
) -> dict:
    """Compare GPU and CPU outputs via canonical vertex identities.

    Uses pre-compaction face data from the diagnostics (WeldResult for
    clip faces, CapResult for cap faces) so that vertex indices refer to
    the original source mesh, not the compacted output array.  Source
    vertices identified by original mesh index; intersection vertices by
    EdgeKey.  Face identity includes canonical vertices, winding relative
    to the clipping plane, and source-face ancestry.
    """
    if gpu_result.status in (
        ClipStatus.ALL_POSITIVE,
        ClipStatus.ALL_NEGATIVE,
        ClipStatus.ALL_COPLANAR,
    ):
        pos_gpu = len(gpu_result.positive_faces)
        pos_cpu = len(cpu_result["pos_faces"])
        neg_gpu = len(gpu_result.negative_faces)
        neg_cpu = len(cpu_result["neg_faces"])
        pos_ok = pos_gpu == pos_cpu
        neg_ok = neg_gpu == neg_cpu
        return {
            "pass": bool(pos_ok and neg_ok),
            "pos_agree": pos_gpu if pos_ok else 0,
            "pos_disagree": 0 if pos_ok else abs(pos_gpu - pos_cpu),
            "neg_agree": neg_gpu if neg_ok else 0,
            "neg_disagree": 0 if neg_ok else abs(neg_gpu - neg_cpu),
            # No plane cut happens for these statuses, so there are no caps
            # to compare -- vacuously equivalent/strict-matching.
            "pos_cap_equivalent": True,
            "neg_cap_equivalent": True,
            "pos_cap_strict": True,
            "neg_cap_strict": True,
        }

    diagnostics = gpu_result.diagnostics
    if diagnostics is None or diagnostics.weld_result is None:
        return {
            "pass": False,
            "pos_agree": 0,
            "pos_disagree": 1,
            "neg_agree": 0,
            "neg_disagree": 1,
            "pos_cap_equivalent": False,
            "neg_cap_equivalent": False,
            "pos_cap_strict": False,
            "neg_cap_strict": False,
        }

    weld = diagnostics.weld_result
    gpu_edge_keys = weld.welded_edge_keys

    gpu_precompact = np.concatenate(
        [
            source_vertices.astype(np.float32),
            weld.welded_positions,
        ]
    )

    pos_clip_agree, pos_clip_disagree = _compare_side_canonical(
        gpu_precompact,
        weld.positive_faces,
        weld.positive_ancestry,
        source_vertex_count,
        gpu_edge_keys,
        cpu_result["pos_vertices"],
        cpu_result["pos_faces"],
        cpu_result["pos_ancestry"],
        source_vertex_count,
        cpu_result["pos_edge_key_map"],
        plane_normal,
    )
    neg_clip_agree, neg_clip_disagree = _compare_side_canonical(
        gpu_precompact,
        weld.negative_faces,
        weld.negative_ancestry,
        source_vertex_count,
        gpu_edge_keys,
        cpu_result["neg_vertices"],
        cpu_result["neg_faces"],
        cpu_result["neg_ancestry"],
        source_vertex_count,
        cpu_result["neg_edge_key_map"],
        plane_normal,
    )

    cap = diagnostics.cap_result
    gpu_pos_caps = (
        cap.positive_cap_faces
        if cap is not None and cap.valid
        else np.zeros((0, 3), dtype=np.uint32)
    )
    gpu_neg_caps = (
        cap.negative_cap_faces
        if cap is not None and cap.valid
        else np.zeros((0, 3), dtype=np.uint32)
    )

    # GPU loops come from the pipeline's own boundary-loop extraction
    # (diagnostics.loop_result), not from translating CPU loops.  Both
    # sides share one set of loops in the pre-compaction vertex space --
    # the same loops that triangulate_caps used.
    loop_result = diagnostics.loop_result
    gpu_loops = (
        list(loop_result.loops) if loop_result is not None and loop_result.valid else []
    )
    negative_loop_result = diagnostics.negative_loop_result or loop_result
    gpu_neg_loops = (
        list(negative_loop_result.loops)
        if negative_loop_result is not None and negative_loop_result.valid
        else []
    )

    pos_cap_result = compare_caps_equivalent(
        gpu_precompact,
        gpu_pos_caps,
        source_vertex_count,
        gpu_edge_keys,
        cpu_result["pos_vertices"],
        cpu_result["pos_cap_faces"],
        source_vertex_count,
        cpu_result["pos_edge_key_map"],
        plane_normal,
        gpu_loops,
        cpu_result["pos_loops"],
        expected_winding_sign=1,
    )
    neg_cap_result = compare_caps_equivalent(
        gpu_precompact,
        gpu_neg_caps,
        source_vertex_count,
        gpu_edge_keys,
        cpu_result["neg_vertices"],
        cpu_result["neg_cap_faces"],
        source_vertex_count,
        cpu_result["neg_edge_key_map"],
        plane_normal,
        gpu_neg_loops,
        cpu_result["neg_loops"],
        expected_winding_sign=-1,
    )

    return {
        "pass": bool(
            pos_clip_disagree == 0
            and neg_clip_disagree == 0
            and pos_cap_result["equivalent"]
            and neg_cap_result["equivalent"]
        ),
        "pos_agree": pos_clip_agree,
        "pos_disagree": pos_clip_disagree,
        "neg_agree": neg_clip_agree,
        "neg_disagree": neg_clip_disagree,
        "pos_cap_equivalent": pos_cap_result["equivalent"],
        "neg_cap_equivalent": neg_cap_result["equivalent"],
        "pos_cap_strict": pos_cap_result["strict_match"],
        "neg_cap_strict": neg_cap_result["strict_match"],
    }


def check_cap_winding(gpu_result: ClipPipelineResult, plane_normal: np.ndarray) -> bool:
    """Verify cap triangles (ancestry == 0xFFFFFFFF) wind toward the
    expected side of the clipping plane."""
    plane_n = np.asarray(plane_normal, dtype=np.float64)

    for verts, faces, ancestry, expected_sign in [
        (
            gpu_result.positive_vertices,
            gpu_result.positive_faces,
            gpu_result.positive_ancestry,
            1,
        ),
        (
            gpu_result.negative_vertices,
            gpu_result.negative_faces,
            gpu_result.negative_ancestry,
            -1,
        ),
    ]:
        if len(faces) == 0:
            continue
        cap_mask = ancestry == 0xFFFFFFFF
        cap_faces = faces[cap_mask]
        for f in cap_faces:
            tri = verts[f].astype(np.float64)
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            nl = np.linalg.norm(normal)
            if nl < 1e-6:
                continue
            normal = normal / nl
            if expected_sign * np.dot(normal, plane_n) < -0.01:
                return False
    return True


def check_criteria(
    gpu_result: ClipPipelineResult,
    cpu_result: dict | None,
    plane_normal: np.ndarray,
    source_vertex_count: int = 0,
    source_vertices: np.ndarray | None = None,
) -> dict:
    """Check all per-clip contract criteria for a single (mesh, plane) pair."""
    results: dict[str, bool] = {}

    # Criterion 1: classification -- checked implicitly through topology
    # parity below (a classification mismatch necessarily shows up as a
    # face-set disagreement).
    results["classification"] = True

    if gpu_result.status == ClipStatus.TOPOLOGY_ERROR:
        # Non-manifold boundary input: extract_cap_f32 degrades gracefully
        # rather than raising, so there is no CPU cap topology to compare --
        # a GPU-reported topology error on this input is treated as
        # concordant with "the CPU oracle has no valid closed-loop topology
        # to offer either."
        results["topology"] = True
    elif gpu_result.status.is_success:
        if cpu_result is None:
            # GPU succeeded but the CPU oracle raised -- a real disagreement,
            # not a case to silently wave through.
            results["topology"] = False
        else:
            results["topology"] = check_topology_parity(
                gpu_result,
                cpu_result,
                plane_normal,
                source_vertex_count,
                source_vertices
                if source_vertices is not None
                else np.zeros((0, 3), dtype=np.float32),
            )["pass"]
    else:
        # CAPACITY_EXCEEDED / INVALID_INPUT / UNSUPPORTED_TOPOLOGY: no CPU
        # comparison is defined for these statuses.
        results["topology"] = True

    # Criterion 3: finite coordinates.
    results["finite_coords"] = bool(
        np.all(np.isfinite(gpu_result.positive_vertices))
        and np.all(np.isfinite(gpu_result.negative_vertices))
    )

    # Criterion 4: valid indices.
    pos_valid = len(
        gpu_result.positive_faces
    ) == 0 or gpu_result.positive_faces.max() < len(gpu_result.positive_vertices)
    neg_valid = len(
        gpu_result.negative_faces
    ) == 0 or gpu_result.negative_faces.max() < len(gpu_result.negative_vertices)
    results["valid_indices"] = bool(pos_valid and neg_valid)

    # Criterion 5: boundary loops closed.
    results["loops_closed"] = gpu_result.status != ClipStatus.TOPOLOGY_ERROR

    # Criterion 6: cap winding.
    results["caps_wound"] = check_cap_winding(gpu_result, plane_normal)

    return results
