"""Cap-triangulation equivalence checking.

This file tests a *standalone* geometric equivalence checker for cap
triangulations: given a boundary loop (the ring of vertices bounding a
clip-plane cap) and a candidate set of triangle faces, decide whether
the faces form a valid triangulation of that loop. Two different valid
triangulations of the *same* loop (e.g. a convex quad split along
either diagonal) are both accepted -- that is the "equivalence" this
module is about.

The live gate evaluator (`scripts/evaluate_gpu_gate.py`) uses
`_compare_caps_equivalent()` for the gating decision, which delegates
per-loop geometric validation to `_check_side_loop_equivalence()`.
`check_triangulation_equivalence` below is a self-contained
reimplementation of the same per-loop equivalence properties — a
standalone regression oracle for the invariants that the gate enforces
(Amendment A1 to the clipping pipeline contract). The evaluator module
itself is loaded lazily via `importlib` (see
`TestNegativeControlIndependentLoops.test_loop_identity_rejected_by_live_evaluator`)
because importing it pulls in `chitin.gpu` / `wgpu` (~15s), so it is
loaded only where the test targets the live loop-matching path.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

_GATE_SCRIPT = Path(__file__).parent.parent / "scripts" / "evaluate_gpu_gate.py"


def _load_gate_evaluator():
    """Load `scripts/evaluate_gpu_gate.py` lazily (~15s: pulls in chitin.gpu
    / wgpu), for tests that need the live `_compare_caps_equivalent()`."""
    spec = importlib.util.spec_from_file_location("evaluate_gpu_gate", _GATE_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Standalone equivalence checker
# ---------------------------------------------------------------------------


def _tri_area(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))


def _polygon_normal_vector(loop_verts: np.ndarray) -> np.ndarray:
    """Newell's method: a vector whose direction/magnitude encode the
    polygon's winding and twice its area."""
    total = np.zeros(3)
    n = len(loop_verts)
    for i in range(n):
        total += np.cross(loop_verts[i], loop_verts[(i + 1) % n])
    return total


def _polygon_area_3d(loop_verts: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(_polygon_normal_vector(loop_verts)))


def _loop_winding_sign(loop_verts: np.ndarray, plane_normal: np.ndarray) -> int:
    s = float(np.dot(_polygon_normal_vector(loop_verts), plane_normal))
    if abs(s) < 1e-12:
        return 0
    return 1 if s > 0 else -1


def _project_2d(points: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """Drop the axis most aligned with plane_normal, project the rest."""
    drop_axis = int(np.argmax(np.abs(plane_normal)))
    keep_axes = [a for a in range(3) if a != drop_axis]
    return points[:, keep_axes]


def _point_on_segment_2d(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float = 1e-9
) -> bool:
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if abs(cross) > tol:
        return False
    dot = (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])
    if dot < -tol:
        return False
    sq_len = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
    return dot <= sq_len + tol


def _point_in_polygon_2d(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test; boundary counts as inside."""
    x, y = float(point[0]), float(point[1])
    n = len(polygon)
    inside = False
    for i in range(n):
        x1, y1 = float(polygon[i][0]), float(polygon[i][1])
        x2, y2 = float(polygon[(i + 1) % n][0]), float(polygon[(i + 1) % n][1])
        if _point_on_segment_2d(
            np.array([x, y]), np.array([x1, y1]), np.array([x2, y2])
        ):
            return True
        if (y1 > y) != (y2 > y):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x_intersect > x:
                inside = not inside
    return inside


def _orientation_2d(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> int:
    val = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    if abs(val) < 1e-9:
        return 0
    return 1 if val > 0 else -1


def _segments_cross(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> bool:
    """True if segment p1-p2 properly crosses segment p3-p4 (interior
    intersection). Shared/touching endpoints do not count as crossing."""
    o1 = _orientation_2d(p1, p2, p3)
    o2 = _orientation_2d(p1, p2, p4)
    o3 = _orientation_2d(p3, p4, p1)
    o4 = _orientation_2d(p3, p4, p2)
    if 0 in (o1, o2, o3, o4):
        return False
    return o1 != o2 and o3 != o4


def check_triangulation_equivalence(
    loop: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    plane_normal: np.ndarray,
    area_rtol: float = 1e-4,
    expected_winding_sign: int = 0,
) -> bool:
    """Check that `faces` is a valid triangulation of the boundary `loop`.

    Validates, in order: (1) vertex membership, (2) face count == n-2,
    (3) nondegeneracy, (4)/(5) boundary-edge/diagonal edge-multiplicity,
    (8) consistent winding with expected polarity, (6)/(7) no crossing
    diagonals/boundary edges, (9) triangle centroids inside the polygon,
    (10) area sum.

    `expected_winding_sign`: +1 for positive-half caps (normals align
    with plane_normal), -1 for negative-half, 0 for self-consistency
    only.
    """
    loop = np.asarray(loop, dtype=np.int64).reshape(-1)
    faces = (
        np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if len(faces)
        else (np.zeros((0, 3), dtype=np.int64))
    )
    plane_normal = np.asarray(plane_normal, dtype=np.float64)
    n = len(loop)

    if n == 0:
        return len(faces) == 0
    if n < 3:
        return False

    loop_set = {int(v) for v in loop}

    # (1) vertex membership
    if len(faces) and not set(faces.flatten().tolist()).issubset(loop_set):
        return False

    # (2) face count
    if len(faces) != n - 2:
        return False

    # Scale-relative degeneracy threshold (same as gate evaluator):
    # d² * 1e-5 where d is the loop bounding-box diagonal.
    loop_verts = verts[loop].astype(np.float64)
    bbox_diag = float(np.linalg.norm(loop_verts.max(axis=0) - loop_verts.min(axis=0)))
    degen_threshold = 0.5 * bbox_diag * bbox_diag * 1e-5

    # (3) nondegeneracy
    tri_areas = []
    for f in faces:
        area = _tri_area(verts[f[0]], verts[f[1]], verts[f[2]])
        if area < degen_threshold:
            return False
        tri_areas.append(area)

    # (4)/(5) edge topology
    boundary_edges = {
        frozenset((int(loop[i]), int(loop[(i + 1) % n]))) for i in range(n)
    }
    directed_counts: dict[tuple[int, int], int] = {}
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            key = (int(a), int(b))
            directed_counts[key] = directed_counts.get(key, 0) + 1
    undirected_counts: dict[frozenset, int] = {}
    for (a, b), c in directed_counts.items():
        key = frozenset((a, b))
        undirected_counts[key] = undirected_counts.get(key, 0) + c

    for edge in boundary_edges:
        if undirected_counts.get(edge, 0) != 1:
            return False

    diagonal_edges = set(undirected_counts) - boundary_edges
    for edge in diagonal_edges:
        if undirected_counts[edge] != 2:
            return False
        a, b = tuple(edge)
        if directed_counts.get((a, b), 0) != 1 or directed_counts.get((b, a), 0) != 1:
            return False

    if set(undirected_counts) != boundary_edges | diagonal_edges:
        return False

    # (8) consistent winding relative to expected side
    # When expected_winding_sign is nonzero, triangles must wind that way.
    # When 0, fall back to self-consistency against the loop's own winding.
    if expected_winding_sign != 0:
        required_positive = expected_winding_sign > 0
    else:
        loop_winding = _loop_winding_sign(verts[loop], plane_normal)
        if loop_winding == 0:
            return False
        required_positive = loop_winding > 0
    for f in faces:
        n_hat = np.cross(verts[f[1]] - verts[f[0]], verts[f[2]] - verts[f[0]])
        s = float(np.dot(n_hat, plane_normal))
        if abs(s) < 1e-12:
            return False
        if (s > 0) != required_positive:
            return False

    # (6)/(7) no crossing diagonals, no diagonal-crosses-boundary
    proj_loop = _project_2d(verts[loop], plane_normal)
    idx_to_proj = {int(v): proj_loop[i] for i, v in enumerate(loop)}
    diag_list = [tuple(e) for e in diagonal_edges]

    for i in range(len(diag_list)):
        for j in range(i + 1, len(diag_list)):
            e1, e2 = diag_list[i], diag_list[j]
            if set(e1) & set(e2):
                continue
            if _segments_cross(
                idx_to_proj[e1[0]],
                idx_to_proj[e1[1]],
                idx_to_proj[e2[0]],
                idx_to_proj[e2[1]],
            ):
                return False

    for diag in diag_list:
        for i in range(n):
            b_edge = (int(loop[i]), int(loop[(i + 1) % n]))
            if set(diag) & set(b_edge):
                continue
            if _segments_cross(
                idx_to_proj[diag[0]],
                idx_to_proj[diag[1]],
                idx_to_proj[b_edge[0]],
                idx_to_proj[b_edge[1]],
            ):
                return False

    # (9) centroid inside boundary polygon
    for f in faces:
        centroid_3d = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        centroid_2d = _project_2d(centroid_3d.reshape(1, 3), plane_normal)[0]
        if not _point_in_polygon_2d(centroid_2d, proj_loop):
            return False

    # (10) area sum
    poly_area = _polygon_area_3d(verts[loop])
    if poly_area < 1e-12:
        return False
    total_tri_area = sum(tri_areas)
    return abs(total_tri_area - poly_area) <= area_rtol * poly_area


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------


def _make_quad_loop(vertices: np.ndarray) -> np.ndarray:
    """Boundary loop indices for a 4-vertex quad, in ring order."""
    assert len(vertices) == 4
    return np.arange(4, dtype=np.int64)


def _make_concave_loop(vertices: np.ndarray) -> np.ndarray:
    """Boundary loop indices for a 5+-vertex (possibly concave) polygon,
    in ring order."""
    assert len(vertices) >= 5
    return np.arange(len(vertices), dtype=np.int64)


def _triangulate_quad_diag02(loop: np.ndarray) -> np.ndarray:
    """Split a 4-vertex loop [a, b, c, d] along diagonal a-c."""
    a, b, c, d = (int(v) for v in loop)
    return np.array([[a, b, c], [a, c, d]], dtype=np.int64)


def _triangulate_quad_diag13(loop: np.ndarray) -> np.ndarray:
    """Split a 4-vertex loop [a, b, c, d] along diagonal b-d."""
    a, b, c, d = (int(v) for v in loop)
    return np.array([[a, b, d], [b, c, d]], dtype=np.int64)


def _regular_polygon(n: int, radius: float = 1.0) -> np.ndarray:
    """CCW-ordered regular n-gon in the XY plane, centered at the origin."""
    angles = [2.0 * math.pi * k / n for k in range(n)]
    return np.array([[radius * math.cos(a), radius * math.sin(a), 0.0] for a in angles])


_SQUARE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
_PLANE_Z = np.array([0.0, 0.0, 1.0])

# L-shaped hexagon: (0,0) (2,0) (2,1) (1,1) (1,2) (0,2), CCW, reflex at index 3.
_L_HEXAGON = np.array(
    [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ]
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCapTriangulationEquivalence:
    # -- Equivalence cases (should pass) ---------------------------------

    def test_convex_quad_opposite_diagonals(self):
        """Two valid triangulations of a convex quad using opposite
        diagonals are equivalent."""
        loop = _make_quad_loop(_SQUARE)
        faces_02 = _triangulate_quad_diag02(loop)
        faces_13 = _triangulate_quad_diag13(loop)

        assert check_triangulation_equivalence(loop, _SQUARE, faces_02, _PLANE_Z)
        assert check_triangulation_equivalence(loop, _SQUARE, faces_13, _PLANE_Z)

    def test_concave_polygon_two_triangulations(self):
        """A concave (L-shaped) polygon with two valid ear-clip orderings
        produces equivalent results."""
        loop = _make_concave_loop(_L_HEXAGON)

        # Fan from vertex 0: diagonals 0-2, 0-3, 0-4.
        fan_from_0 = np.array(
            [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]], dtype=np.int64
        )
        # Fan from vertex 3 (the reflex vertex sees 0, 1, 5): diagonals
        # 3-0, 3-1, 3-5.
        fan_from_3 = np.array(
            [[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]], dtype=np.int64
        )

        assert check_triangulation_equivalence(loop, _L_HEXAGON, fan_from_0, _PLANE_Z)
        assert check_triangulation_equivalence(loop, _L_HEXAGON, fan_from_3, _PLANE_Z)

    def test_triangle_identical(self):
        """A single triangle (n=3, n-2=1) is trivially equivalent to
        itself."""
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        loop = np.array([0, 1, 2], dtype=np.int64)
        faces = np.array([[0, 1, 2]], dtype=np.int64)

        assert check_triangulation_equivalence(loop, verts, faces, _PLANE_Z)
        assert check_triangulation_equivalence(loop, verts, faces, _PLANE_Z)

    def test_pentagon_two_valid(self):
        """A regular pentagon with two different valid (fan)
        triangulations."""
        pentagon = _regular_polygon(5)
        loop = _make_concave_loop(pentagon)

        fan_from_0 = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]], dtype=np.int64)
        fan_from_1 = np.array([[1, 2, 3], [1, 3, 4], [1, 4, 0]], dtype=np.int64)

        assert check_triangulation_equivalence(loop, pentagon, fan_from_0, _PLANE_Z)
        assert check_triangulation_equivalence(loop, pentagon, fan_from_1, _PLANE_Z)

    # -- Failure cases (should fail) --------------------------------------

    def test_missing_triangle(self):
        """Fewer than n-2 triangles for an n-vertex loop: not enough
        coverage."""
        loop = _make_quad_loop(_SQUARE)
        faces = np.array([[0, 1, 2]], dtype=np.int64)  # only 1, need 2

        assert not check_triangulation_equivalence(loop, _SQUARE, faces, _PLANE_Z)

    def test_extra_triangle(self):
        """More than n-2 triangles for an n-vertex loop: too many."""
        loop = _make_quad_loop(_SQUARE)
        faces = np.array([[0, 1, 2], [0, 2, 3], [1, 2, 3]], dtype=np.int64)  # 3, need 2

        assert not check_triangulation_equivalence(loop, _SQUARE, faces, _PLANE_Z)

    def test_degenerate_triangle(self):
        """A triangulation containing a zero-area triangle (three
        collinear vertices)."""
        verts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
        )
        loop = np.array([0, 1, 2, 3], dtype=np.int64)
        # Vertices 0, 1, 2 are collinear -> triangle [0, 1, 2] is degenerate.
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

        assert not check_triangulation_equivalence(loop, verts, faces, _PLANE_Z)

    def test_crossing_diagonals(self):
        """A concave polygon where the attempted diagonals cut through
        the reflex notch -- combinatorially balanced (every boundary
        edge once, every diagonal twice, oppositely) but geometrically
        invalid."""
        loop = _make_concave_loop(_L_HEXAGON)
        # Corner ears (0,1,2), (2,3,4), (4,5,0) plus a middle triangle
        # (0,2,4): edge (2,4) cuts straight through the reflex notch at
        # vertex 3, so triangle (2,3,4)'s centroid falls outside the
        # polygon even though every edge count balances.
        faces = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 0], [0, 2, 4]], dtype=np.int64)

        assert not check_triangulation_equivalence(loop, _L_HEXAGON, faces, _PLANE_Z)

    def test_reversed_winding(self):
        """All triangles wound CW instead of CCW relative to the plane
        normal."""
        loop = _make_quad_loop(_SQUARE)
        faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64)  # reversed

        assert not check_triangulation_equivalence(loop, _SQUARE, faces, _PLANE_Z)

    def test_reversed_winding_with_expected_sign(self):
        """Reversed winding passes when expected_winding_sign matches it,
        fails when it doesn't."""
        loop = _make_quad_loop(_SQUARE)
        # CW triangles — normal dots plane_normal < 0
        faces_cw = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
        # CCW triangles — normal dots plane_normal > 0
        faces_ccw = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

        assert check_triangulation_equivalence(
            loop, _SQUARE, faces_cw, _PLANE_Z, expected_winding_sign=-1
        )
        assert not check_triangulation_equivalence(
            loop, _SQUARE, faces_cw, _PLANE_Z, expected_winding_sign=1
        )
        assert check_triangulation_equivalence(
            loop, _SQUARE, faces_ccw, _PLANE_Z, expected_winding_sign=1
        )
        assert not check_triangulation_equivalence(
            loop, _SQUARE, faces_ccw, _PLANE_Z, expected_winding_sign=-1
        )

    def test_same_area_wrong_boundary(self):
        """Triangles sum to the correct area but reference an entirely
        different loop's vertices."""
        other_square = _SQUARE + np.array([5.0, 5.0, 0.0])
        verts = np.concatenate([_SQUARE, other_square], axis=0)
        loop = _make_quad_loop(_SQUARE)  # loop = [0, 1, 2, 3]
        # Same-area quad, but built from the *other* square's vertices (4-7).
        faces = np.array([[4, 5, 6], [4, 6, 7]], dtype=np.int64)

        assert not check_triangulation_equivalence(loop, verts, faces, _PLANE_Z)

    def test_vertex_outside_loop(self):
        """A cap triangle references a vertex not in the boundary loop."""
        verts = np.concatenate(
            [_SQUARE, np.array([[0.5, 0.5, 1.0]])], axis=0
        )  # vertex 4 is off-plane and outside the loop
        loop = _make_quad_loop(_SQUARE)
        faces = np.array([[0, 1, 4], [0, 4, 3]], dtype=np.int64)

        assert not check_triangulation_equivalence(loop, verts, faces, _PLANE_Z)

    # -- Structural tests ---------------------------------------------------

    def test_empty_loops_both_sides(self):
        """No loops on either side is trivially equivalent."""
        loop = np.zeros((0,), dtype=np.int64)
        faces = np.zeros((0, 3), dtype=np.int64)
        verts = np.zeros((0, 3), dtype=np.float64)

        assert check_triangulation_equivalence(loop, verts, faces, _PLANE_Z)

    def test_multiple_independent_loops(self):
        """Two separate loops, each independently equivalent."""
        other_square = _SQUARE + np.array([10.0, 0.0, 0.0])
        verts = np.concatenate([_SQUARE, other_square], axis=0)

        loop_a = np.array([0, 1, 2, 3], dtype=np.int64)
        loop_b = np.array([4, 5, 6, 7], dtype=np.int64)
        faces_a = _triangulate_quad_diag02(loop_a)
        faces_b = _triangulate_quad_diag13(loop_b)

        assert check_triangulation_equivalence(loop_a, verts, faces_a, _PLANE_Z)
        assert check_triangulation_equivalence(loop_b, verts, faces_b, _PLANE_Z)

    def test_determinism_ten_runs(self):
        """Same inputs produce identical equivalence results across 10
        runs."""
        loop = _make_quad_loop(_SQUARE)
        faces = _triangulate_quad_diag02(loop)

        results = [
            check_triangulation_equivalence(loop, _SQUARE, faces, _PLANE_Z)
            for _ in range(10)
        ]

        assert all(results)
        assert len(set(results)) == 1


class TestNegativeControlIndependentLoops:
    """Verify equivalence is non-circular: GPU and CPU loops are compared
    independently. A triangulation valid for one boundary must fail when
    checked against a different boundary's edge connectivity."""

    def test_corrupted_boundary_fails_edge_coverage(self):
        """A triangulation built for one boundary cycle fails the
        boundary-edge-coverage check when evaluated against a different
        boundary cycle."""
        valid_loop = np.array([0, 1, 2, 3], dtype=np.int64)
        valid_faces = _triangulate_quad_diag02(valid_loop)
        assert check_triangulation_equivalence(
            valid_loop, _SQUARE, valid_faces, _PLANE_Z
        )

        # Same faces evaluated against a different boundary (swapped 1,2)
        corrupted_loop = np.array([0, 2, 1, 3], dtype=np.int64)
        assert not check_triangulation_equivalence(
            corrupted_loop, _SQUARE, valid_faces, _PLANE_Z
        )

    def test_valid_faces_wrong_loop_same_vertex_set(self):
        """Triangulation uses the correct vertices but against a loop with
        different edge connectivity. Boundary-edge coverage fails."""
        pent = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.5, 1.0, 0.0],
                [0.5, 1.5, 0.0],
                [-0.5, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        loop_a = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        faces_a = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]], dtype=np.int64)
        assert check_triangulation_equivalence(loop_a, pent, faces_a, _PLANE_Z)

        # Same vertices, different cyclic order
        loop_b = np.array([0, 2, 4, 1, 3], dtype=np.int64)
        assert not check_triangulation_equivalence(loop_b, pent, faces_a, _PLANE_Z)

    def test_loop_identity_rejected_by_live_evaluator(self):
        """Two loops over the same vertex set, each independently a valid
        simple polygon with a valid triangulation, but canonically
        different cyclic orders (a notch on the left vs. a notch on the
        right of the same square-plus-center point set). Neither the
        standalone checker nor the live gate evaluator's
        `_compare_caps_equivalent()` may treat them as equivalent."""
        verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        # Notched on the left side (reflex at vertex 4).
        loop_a = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        faces_a = np.array([[4, 0, 1], [4, 1, 2], [4, 2, 3]], dtype=np.int64)
        # Notched on the right side (reflex at vertex 4). Same vertex set,
        # different cyclic order -- not merely a reversal of loop_a.
        loop_b = np.array([0, 1, 4, 2, 3], dtype=np.int64)
        faces_b = np.array([[4, 2, 3], [4, 3, 0], [4, 0, 1]], dtype=np.int64)

        assert check_triangulation_equivalence(loop_a, verts, faces_a, _PLANE_Z)
        assert check_triangulation_equivalence(loop_b, verts, faces_b, _PLANE_Z)

        # The two loops are not the same cycle: their lex-smallest
        # rotations differ (0,1,2,3,4) vs (0,1,4,2,3).
        assert tuple(loop_a.tolist()) != tuple(loop_b.tolist())

        gate = _load_gate_evaluator()
        empty_edge_keys = np.zeros((0, 2), dtype=np.int64)

        result = gate._compare_caps_equivalent(
            verts,
            faces_a.astype(np.int64),
            5,
            empty_edge_keys,
            verts,
            faces_b.astype(np.int64),
            5,
            empty_edge_keys,
            _PLANE_Z,
            [loop_a.astype(np.int64)],
            [loop_b.astype(np.int64)],
        )
        assert result["equivalent"] is False
        assert any(
            detail.get("error") == "unmatched_loop" for detail in result["details"]
        )

        # Positive control: the same loop/faces on both sides must be
        # accepted, so the rejection above is loop identity, not a
        # broken harness.
        control = gate._compare_caps_equivalent(
            verts,
            faces_a.astype(np.int64),
            5,
            empty_edge_keys,
            verts,
            faces_a.astype(np.int64),
            5,
            empty_edge_keys,
            _PLANE_Z,
            [loop_a.astype(np.int64)],
            [loop_a.astype(np.int64)],
        )
        assert control["equivalent"] is True


def test_live_evaluator_distinguishes_thin_from_degenerate():
    gate = _load_gate_evaluator()
    loop = np.arange(3)
    faces = np.array([[0, 1, 2]])
    thin = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1e-7, 0.0]])
    ok, details = gate._check_side_loop_equivalence(loop, faces, thin, _PLANE_Z, 1)
    assert ok
    assert details["low_margin_triangle_count"] == 1
    thin[2, 1] = 0
    ok, details = gate._check_side_loop_equivalence(loop, faces, thin, _PLANE_Z, 1)
    assert not ok
    assert not details["nondegenerate"]


def test_live_evaluator_matches_reversed_boundary():
    gate = _load_gate_evaluator()
    loop = np.array([0, 1, 2, 3])
    faces = _triangulate_quad_diag02(loop)
    keys = np.zeros((0, 2), dtype=np.uint32)
    result = gate._compare_caps_equivalent(
        _SQUARE,
        faces,
        4,
        keys,
        _SQUARE,
        faces,
        4,
        keys,
        _PLANE_Z,
        [loop],
        [loop[::-1]],
        expected_winding_sign=1,
    )
    assert result["equivalent"]
    reversed_faces = faces[:, [0, 2, 1]]
    result = gate._compare_caps_equivalent(
        _SQUARE,
        reversed_faces,
        4,
        keys,
        _SQUARE,
        faces,
        4,
        keys,
        _PLANE_Z,
        [loop],
        [loop[::-1]],
        expected_winding_sign=1,
    )
    assert not result["equivalent"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
