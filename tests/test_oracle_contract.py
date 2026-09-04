"""Oracle on-plane contract tests (chitin #123).

Verifies that compare_oracle() correctly excuses on-plane convention
differences: when C++ CoACD reports side=0 (on-plane), any f32
classification is valid and does not count as a disagreement — but only
within the f32 distance guard bound (chitin #124).
"""

import numpy as np

from chitin.coacd_trace import TracedClip, TracedPlane
from chitin.coacd_trace_replay import compare_oracle


def _synthetic_clip(
    vertices: np.ndarray,
    oracle_sides: np.ndarray,
    plane_normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    plane_offset: float = 0.0,
) -> TracedClip:
    """Build a minimal TracedClip for oracle contract testing."""
    n = len(vertices)
    faces = (
        np.zeros((0, 3), dtype=np.int32)
        if n < 3
        else np.array([[0, 1, 2]], dtype=np.int32)
    )
    return TracedClip(
        component_id=0,
        plane=TracedPlane(
            a=plane_normal[0],
            b=plane_normal[1],
            c=plane_normal[2],
            d=plane_offset,
            method="clip",
            index=-1,
        ),
        pos_verts=0,
        pos_faces=0,
        neg_verts=0,
        neg_faces=0,
        intersection_count=0,
        input_vertices=vertices.astype(np.float64),
        input_faces=faces,
        oracle_sides=oracle_sides.astype(np.int8),
    )


class TestOracleOnPlaneContract:
    """On-plane vertices (oracle_side=0) excused only within f32 error bound."""

    # Markers scaled to 0.1 (not 1.0): classify_plane_f32's default policy
    # (DEFAULT_POLICY, grid_bits=20, with ambiguity fallback) resolves signs
    # from a grid quantized to the clip's own vertex extent. A near-plane
    # offset only survives that quantization — and stays within the f32
    # distance-guard bound (~9.5e-7 at unit scale) — when the clip's extent
    # is kept small; 0.1 markers with a 5e-7 offset satisfy both.

    def test_near_plane_oracle_zero_excused(self):
        """oracle=0, vertex near plane (z=5e-7) -> excused."""
        verts = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1], [0.0, 0.0, 5e-7]])
        oracle = np.array([1, -1, 0], dtype=np.int8)
        result = compare_oracle(_synthetic_clip(verts, oracle), clip_index=0)
        assert result is not None
        assert result.on_plane_excused == 1
        assert result.num_disagree == 0
        assert result.agreement_rate == 1.0

    def test_near_plane_negative_excused(self):
        """oracle=0, vertex near plane (z=-5e-7) -> f32=-1, excused."""
        verts = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1], [0.0, 0.0, -5e-7]])
        oracle = np.array([1, -1, 0], dtype=np.int8)
        result = compare_oracle(_synthetic_clip(verts, oracle), clip_index=0)
        assert result is not None
        assert result.on_plane_excused == 1
        assert result.num_disagree == 0

    def test_distant_oracle_zero_not_excused(self):
        """oracle=0 at z=1.0 from plane -> not excused (distance guard)."""
        verts = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0], [0.0, 0.0, 1.0]])
        oracle = np.array([1, -1, 0], dtype=np.int8)
        result = compare_oracle(_synthetic_clip(verts, oracle), clip_index=0)
        assert result is not None
        assert result.on_plane_excused == 0
        assert result.num_disagree == 1

    def test_exact_zero_agree(self):
        """oracle=0, f32=0 (vertex at z=0.0 exactly) -> exact agree."""
        verts = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
        oracle = np.array([1, -1, 0], dtype=np.int8)
        result = compare_oracle(_synthetic_clip(verts, oracle), clip_index=0)
        assert result is not None
        # z=0.0 exactly: both oracle and f32 return 0 -> exact agree
        assert result.num_disagree == 0

    def test_genuine_disagree_both_nonzero(self):
        """oracle=+1, f32=-1 -> genuine disagree, never excused."""
        verts = np.array(
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, -2.0],
                [0.0, 0.0, 2.0],
            ]
        )
        oracle = np.array([1, -1, -1], dtype=np.int8)
        result = compare_oracle(_synthetic_clip(verts, oracle), clip_index=0)
        assert result is not None
        assert result.on_plane_excused == 0
        assert result.num_disagree == 1
        assert result.agreement_rate < 1.0

    def test_strict_vs_excused_rates(self):
        """strict_agreement_rate < agreement_rate when near-plane excused."""
        verts = np.array(
            [
                [0.0, 0.0, 0.1],
                [0.0, 0.0, -0.1],
                [0.0, 0.0, 5e-7],
                [0.0, 0.0, -5e-7],
            ]
        )
        oracle = np.array([1, -1, 0, 0], dtype=np.int8)
        result = compare_oracle(_synthetic_clip(verts, oracle), clip_index=0)
        assert result is not None
        assert result.agreement_rate >= result.strict_agreement_rate
        assert result.num_disagree == 0
        assert result.on_plane_excused == 2

    def test_all_near_plane_excused(self):
        """All oracle_side=0 near plane -> all excused, 100% rate."""
        verts = np.array(
            [
                [0.0, 0.0, 1e-10],
                [0.0, 0.0, -1e-10],
                [0.0, 0.0, 1e-15],
            ]
        )
        oracle = np.array([0, 0, 0], dtype=np.int8)
        result = compare_oracle(_synthetic_clip(verts, oracle), clip_index=0)
        assert result is not None
        assert result.num_disagree == 0
        assert result.agreement_rate == 1.0

    def test_mixed_excused_genuine_and_distant(self):
        """Near-plane excused + distant oracle=0 genuine + real disagree."""
        verts = np.array(
            [
                [0.0, 0.0, 0.1],  # clearly +1, oracle agrees
                [0.0, 0.0, -0.1],  # clearly -1, oracle agrees
                [0.0, 0.0, 5e-7],  # near-plane, oracle=0 -> excused
                [0.0, 0.0, 0.05],  # distant, oracle=0 -> NOT excused (guard)
                [0.0, 0.0, 0.1],  # clearly +1, oracle lies (-1) -> genuine
            ]
        )
        oracle = np.array([1, -1, 0, 0, -1], dtype=np.int8)
        result = compare_oracle(_synthetic_clip(verts, oracle), clip_index=0)
        assert result is not None
        assert result.num_agree == 2
        assert result.on_plane_excused == 1
        assert result.num_disagree == 2  # distant oracle=0 + genuine lie
        assert result.agreement_rate == 3 / 5  # (2 agree + 1 excused) / 5
        assert result.strict_agreement_rate == 2 / 5
