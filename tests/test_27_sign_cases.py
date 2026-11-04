"""Exhaustive tests for all 27 ``{-1, 0, +1}^3`` triangle sign-classification cases.

A single triangle has 3 vertices; each vertex classifies as negative (-1),
on-plane (0), or positive (+1) relative to a clipping plane, giving
``3**3 == 27`` possible sign combinations. This file locks down CPU
(``classify_plane_f32``, ``clip_mesh_f32``) and GPU (``dispatch_clip_pipeline``)
behavior for every one of them, including the fully coplanar (0,0,0) case and
the vertex-on-plane cases.

The expected counts and statuses below were derived by hand from
``_clip_mesh_generic`` (CPU, ``chitin/f32_predicates.py``) and the
``clip_count.wgsl`` shader plus the status-shortcut logic in
``dispatch_clip_pipeline`` (GPU, ``chitin/gpu/clip_pipeline.py``,
``chitin/gpu/pipeline.py``), then confirmed against a real run on this
machine's GPU adapter before being written into assertions.

Notable, non-obvious real behavior locked in here: for a *lone* triangle (no
neighboring triangles to supply the rest of a cut boundary), the 12 "mixed"
sign combinations -- those with both a strictly-positive and a
strictly-negative vertex -- clip to exactly one boundary edge, which by
itself is an *open* chain. ``extract_boundary_loops`` requires every boundary
vertex to have valence exactly 2 (a closed loop), which a single edge can
never satisfy in isolation. So ``dispatch_clip_pipeline`` correctly reports
``ClipStatus.TOPOLOGY_ERROR`` for all 12 mixed combos when the input mesh is
a single isolated triangle -- expected behavior for degenerate single-triangle
input, not a pipeline bug. The CPU per-triangle oracle (``clip_mesh_f32``) has
no such closure requirement and always succeeds.
"""

from __future__ import annotations

from itertools import product
from typing import ClassVar

import numpy as np
import pytest

from chitin.f32_policy import POLICY_0_3_0
from chitin.f32_predicates import classify_plane_f32, clip_mesh_f32

try:
    from chitin.gpu.clip_pipeline import dispatch_clip_pipeline
    from chitin.gpu.pipeline import ClipStatus
    from chitin.gpu.worker import GPUWorker

    GPU_AVAILABLE = GPUWorker.available()
except (ImportError, OSError, RuntimeError):
    GPU_AVAILABLE = False


def make_triangle_mesh(
    signs: tuple[int, int, int],
    plane_normal: np.ndarray | None = None,
    plane_d: float = 0.0,
):
    """Position a single triangle's 3 vertices to realize `signs` relative to
    the plane through `plane_point` with normal `plane_normal`.

    sign -1 places a vertex at z = plane_d - 1.0, sign 0 at z = plane_d,
    sign +1 at z = plane_d + 1.0. Distinct x/y per vertex avoids a degenerate
    (zero-area) triangle. Returns (vertices, faces, plane_point, plane_normal).
    """
    if plane_normal is None:
        plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    z_map = {-1: -1.0, 0: 0.0, 1: 1.0}
    vertices = np.array(
        [
            [0.0, 0.0, z_map[signs[0]] + plane_d],
            [1.0, 0.0, z_map[signs[1]] + plane_d],
            [0.5, 0.866, z_map[signs[2]] + plane_d],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    plane_point = np.array([0.0, 0.0, plane_d], dtype=np.float32)
    return vertices, faces, plane_point, plane_normal.astype(np.float32)


ALL_SIGN_COMBOS: list[tuple[int, int, int]] = list(product([-1, 0, 1], repeat=3))
COMBO_IDS = [f"{s[0]:+d},{s[1]:+d},{s[2]:+d}" for s in ALL_SIGN_COMBOS]


def _expected_cpu_counts(signs: tuple[int, int, int]) -> tuple[int, int, int]:
    """Expected (pos_face_count, intersection_count, boundary_edge_count)
    from `clip_mesh_f32` (positive side only) for a single triangle."""
    s = sorted(signs)
    has_pos = 1 in signs
    has_neg = -1 in signs
    n_zero = signs.count(0)

    if s == [1, 1, 1]:
        return 1, 0, 0
    if s == [-1, -1, -1]:
        return 0, 0, 0
    if s == [0, 0, 0]:
        return 1, 0, 3
    if not has_neg:  # subset of {0, +1} with at least one +1
        return 1, 0, (1 if n_zero == 2 else 0)
    if not has_pos:  # subset of {0, -1} with at least one -1
        return 0, 0, 0
    # mixed: at least one +1 and one -1 present
    if n_zero == 1:
        return 1, 1, 1
    if signs.count(1) == 2:
        return 2, 2, 1
    return 1, 2, 1  # signs.count(-1) == 2


def _expected_gpu_status(signs: tuple[int, int, int]):
    if signs == (0, 0, 0):
        return ClipStatus.OK
    has_pos = 1 in signs
    has_neg = -1 in signs
    if has_pos and has_neg:
        return ClipStatus.TOPOLOGY_ERROR
    if has_neg:
        return ClipStatus.ALL_NEGATIVE
    return ClipStatus.ALL_POSITIVE


@pytest.fixture(scope="module")
def worker():
    if not GPU_AVAILABLE:
        pytest.skip("No GPU adapter available")
    with GPUWorker() as w:
        yield w


class TestClassification:
    """CPU classification (`classify_plane_f32`) assigns the constructed sign
    to every vertex, for all 27 combinations."""

    @pytest.mark.parametrize("signs", ALL_SIGN_COMBOS, ids=COMBO_IDS)
    def test_cpu_classification_matches_construction(self, signs):
        vertices, _faces, plane_point, plane_normal = make_triangle_mesh(signs)
        result = classify_plane_f32(vertices, plane_point, plane_normal, POLICY_0_3_0)
        np.testing.assert_array_equal(result.signs, np.array(signs, dtype=np.int8))

        expected_pos = sum(1 for s in signs if s == 1)
        expected_neg = sum(1 for s in signs if s == -1)
        expected_on_plane = sum(1 for s in signs if s == 0)
        assert result.positive_count == expected_pos
        assert result.negative_count == expected_neg
        assert result.on_plane_count == expected_on_plane


class TestCpuClipCounts:
    """CPU clip oracle (`clip_mesh_f32`) face/intersection/boundary counts for
    all 27 combinations. Runs with no GPU dependency."""

    @pytest.mark.parametrize("signs", ALL_SIGN_COMBOS, ids=COMBO_IDS)
    def test_clip_counts(self, signs):
        vertices, faces, plane_point, plane_normal = make_triangle_mesh(signs)
        clip = clip_mesh_f32(vertices, faces, plane_point, plane_normal, POLICY_0_3_0)

        expected_pos, expected_ixn, expected_bnd = _expected_cpu_counts(signs)
        assert len(clip.faces) == expected_pos, (
            f"signs={signs}: pos_face_count={len(clip.faces)}, expected {expected_pos}"
        )
        assert len(clip.intersection_points) == expected_ixn, (
            f"signs={signs}: intersection_count={len(clip.intersection_points)}, "
            f"expected {expected_ixn}"
        )
        assert len(clip.boundary_edges) == expected_bnd, (
            f"signs={signs}: boundary_edge_count={len(clip.boundary_edges)}, "
            f"expected {expected_bnd}"
        )

        # Every kept face vertex must be finite and every intersection point
        # must actually lie on the plane.
        assert np.all(np.isfinite(clip.vertices))
        if len(clip.intersection_points) > 0:
            on_plane = np.dot(clip.intersection_points - plane_point, plane_normal)
            np.testing.assert_allclose(on_plane, 0.0, atol=1e-4)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU adapter available")
class TestGpuStatus:
    """`dispatch_clip_pipeline` reports the expected `ClipStatus` for every
    one of the 27 combinations, for a single isolated triangle."""

    @pytest.mark.parametrize("signs", ALL_SIGN_COMBOS, ids=COMBO_IDS)
    def test_status(self, signs, worker):
        vertices, faces, plane_point, plane_normal = make_triangle_mesh(signs)
        result = dispatch_clip_pipeline(
            worker, vertices, faces, plane_point, plane_normal
        )
        expected_status = _expected_gpu_status(signs)
        assert result.status == expected_status, (
            f"signs={signs}: status={result.status.name}, "
            f"expected {expected_status.name}"
        )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU adapter available")
class TestGpuTopologyDetail:
    """Detailed topology checks for the GPU pipeline result, grouped by the
    three status outcomes a single isolated triangle can produce."""

    ALL_POSITIVE_SIGNS: ClassVar[list] = [
        s for s in ALL_SIGN_COMBOS if _expected_gpu_status(s).name == "ALL_POSITIVE"
    ]
    ALL_NEGATIVE_SIGNS: ClassVar[list] = [
        s for s in ALL_SIGN_COMBOS if _expected_gpu_status(s).name == "ALL_NEGATIVE"
    ]
    TOPOLOGY_ERROR_SIGNS: ClassVar[list] = [
        s for s in ALL_SIGN_COMBOS if _expected_gpu_status(s).name == "TOPOLOGY_ERROR"
    ]

    @pytest.mark.parametrize(
        "signs",
        ALL_POSITIVE_SIGNS,
        ids=[f"{s[0]:+d},{s[1]:+d},{s[2]:+d}" for s in ALL_POSITIVE_SIGNS],
    )
    def test_all_positive_keeps_whole_triangle(self, signs, worker):
        vertices, faces, plane_point, plane_normal = make_triangle_mesh(signs)
        result = dispatch_clip_pipeline(
            worker, vertices, faces, plane_point, plane_normal
        )
        assert result.status == ClipStatus.ALL_POSITIVE
        assert len(result.positive_faces) == 1
        assert len(result.negative_faces) == 0
        assert len(result.positive_vertices) == 3
        np.testing.assert_array_equal(
            result.positive_ancestry, np.array([0], dtype=np.uint32)
        )
        assert np.all(np.isfinite(result.positive_vertices))

    @pytest.mark.parametrize(
        "signs",
        ALL_NEGATIVE_SIGNS,
        ids=[f"{s[0]:+d},{s[1]:+d},{s[2]:+d}" for s in ALL_NEGATIVE_SIGNS],
    )
    def test_all_negative_keeps_whole_triangle(self, signs, worker):
        vertices, faces, plane_point, plane_normal = make_triangle_mesh(signs)
        result = dispatch_clip_pipeline(
            worker, vertices, faces, plane_point, plane_normal
        )
        assert result.status == ClipStatus.ALL_NEGATIVE
        assert len(result.negative_faces) == 1
        assert len(result.positive_faces) == 0
        assert len(result.negative_vertices) == 3
        np.testing.assert_array_equal(
            result.negative_ancestry, np.array([0], dtype=np.uint32)
        )
        assert np.all(np.isfinite(result.negative_vertices))

    def test_all_coplanar_triangle_caps_both_sides(self, worker):
        """(0,0,0): the triangle's own 3 edges are all on-plane, forming a
        closed boundary loop around itself, so the pipeline both keeps the
        original triangle and triangulates a matching cap on each side."""
        signs = (0, 0, 0)
        vertices, faces, plane_point, plane_normal = make_triangle_mesh(signs)
        result = dispatch_clip_pipeline(
            worker, vertices, faces, plane_point, plane_normal
        )

        assert result.status == ClipStatus.OK
        assert result.metrics.coplanar_face_count == 1
        assert result.metrics.boundary_loop_count == 1
        assert result.metrics.cap_face_count == 2  # one attached cap per half
        # Original triangle + 1 triangulated cap face, on each side.
        assert len(result.positive_faces) == 2
        assert len(result.negative_faces) == 2
        assert np.all(np.isfinite(result.positive_vertices))
        assert np.all(np.isfinite(result.negative_vertices))
        if len(result.positive_faces) > 0:
            assert result.positive_faces.max() < len(result.positive_vertices)
        if len(result.negative_faces) > 0:
            assert result.negative_faces.max() < len(result.negative_vertices)

    @pytest.mark.parametrize(
        "signs",
        TOPOLOGY_ERROR_SIGNS,
        ids=[f"{s[0]:+d},{s[1]:+d},{s[2]:+d}" for s in TOPOLOGY_ERROR_SIGNS],
    )
    def test_mixed_lone_triangle_reports_topology_error(self, signs, worker):
        """A lone triangle with one strictly-positive and one
        strictly-negative vertex always cuts to exactly one boundary edge,
        an open chain `extract_boundary_loops` cannot close without a
        neighboring triangle -- so the pipeline reports TOPOLOGY_ERROR
        rather than fabricating a cap. The CPU oracle has no such
        requirement and still clips cleanly (see TestCpuClipCounts)."""
        vertices, faces, plane_point, plane_normal = make_triangle_mesh(signs)
        result = dispatch_clip_pipeline(
            worker, vertices, faces, plane_point, plane_normal
        )
        assert result.status == ClipStatus.TOPOLOGY_ERROR
        assert len(result.positive_faces) == 0
        assert len(result.negative_faces) == 0

        # The CPU oracle, unlike the GPU pipeline, has no closed-loop
        # requirement and clips this same triangle successfully.
        clip = clip_mesh_f32(vertices, faces, plane_point, plane_normal, POLICY_0_3_0)
        expected_pos, _, _ = _expected_cpu_counts(signs)
        assert len(clip.faces) == expected_pos
