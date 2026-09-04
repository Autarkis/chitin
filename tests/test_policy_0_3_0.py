"""Policy 0.3.0 (#122): canonical IEEE-f32 input contract."""

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from chitin.coacd_trace import CoACDTrace, TracedClip, TracedPlane
from chitin.coacd_trace_replay import compare_oracle, replay_classifications
from chitin.f32_adversarial import (
    PlaneCase,
    evaluate_case,
    evaluate_case_canonical,
)
from chitin.f32_policy import (
    DEFAULT_POLICY,
    POLICY_0_2_0,
    POLICY_0_3_0,
    QuantizationPolicy,
)
from chitin.f32_predicates import (
    canonicalize_inputs_f32,
    categorize_disagreements,
    classify_plane_f32,
    classify_plane_f64,
    diff_classifications,
)

HOLDOUT_FAILURE_DIR = (
    Path(__file__).parent / "fixtures" / "traces" / "holdout_failures_0_2_0"
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
evaluate_holdout = importlib.import_module("evaluate_holdout")


def _holdout_failure_files() -> list[str]:
    manifest = json.loads((HOLDOUT_FAILURE_DIR / "manifest.json").read_text())
    return [entry["file"] for entry in manifest["clips"]]


def _point_and_normal(data):
    raw_normal = data["plane_normal"].astype(np.float64)
    offset = float(data["plane_offset"])
    normal = raw_normal / np.linalg.norm(raw_normal)
    point = normal * (-offset / np.dot(normal, normal))
    return point, normal


class TestCanonicalization:
    def test_round_trip_preserves_finite(self):
        vertices = np.array([[1.5, -2.3, 0.7], [1e10, 1e-10, 0.0]], dtype=np.float64)
        point = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        cv, cp, cn = canonicalize_inputs_f32(vertices, point, normal)
        assert np.all(np.isfinite(cv))
        assert np.all(np.isfinite(cp))
        assert np.all(np.isfinite(cn))
        assert cv.dtype == np.float64
        assert cp.dtype == np.float64
        assert cn.dtype == np.float64

    def test_idempotent(self):
        vertices = np.array([[1.1, 2.2, 3.3]], dtype=np.float64)
        point = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        cv1, cp1, cn1 = canonicalize_inputs_f32(vertices, point, normal)
        cv2, cp2, cn2 = canonicalize_inputs_f32(cv1, cp1, cn1)
        np.testing.assert_array_equal(cv1, cv2)
        np.testing.assert_array_equal(cp1, cp2)
        np.testing.assert_array_equal(cn1, cn2)

    def test_f32_representable_inputs_unchanged(self):
        vertices = np.array([[1.0, 2.0, 3.0]], dtype=np.float32).astype(np.float64)
        point = np.zeros(3, dtype=np.float32).astype(np.float64)
        normal = np.array([1.0, 0.0, 0.0], dtype=np.float32).astype(np.float64)
        cv, cp, cn = canonicalize_inputs_f32(vertices, point, normal)
        np.testing.assert_array_equal(cv, vertices)
        np.testing.assert_array_equal(cp, point)
        np.testing.assert_array_equal(cn, normal)


class TestSourcePrecisionLoss:
    @pytest.mark.parametrize("filename", _holdout_failure_files())
    def test_diagnostic_clips_show_precision_loss_not_arithmetic_error(self, filename):
        with np.load(HOLDOUT_FAILURE_DIR / filename) as data:
            vertices = data["vertices"].astype(np.float64)
            point, normal = _point_and_normal(data)

        result = evaluate_case(PlaneCase(vertices, point, normal, label=filename))
        assert result.num_f32_exact_mismatch > 0
        mismatch = result.f32_exact_mismatch
        assert np.all(result.input_precision_loss[mismatch])
        assert not np.any(result.f32_arithmetic_mismatch[mismatch])

    @pytest.mark.parametrize("filename", _holdout_failure_files())
    def test_canonical_evaluation_agrees_on_diagnostic_clips(self, filename):
        with np.load(HOLDOUT_FAILURE_DIR / filename) as data:
            vertices = data["vertices"].astype(np.float64)
            point, normal = _point_and_normal(data)

        result = evaluate_case_canonical(
            PlaneCase(vertices, point, normal, label=filename)
        )
        assert result.num_f32_exact_mismatch == 0
        assert not np.any(result.f32_arithmetic_mismatch)
        assert not np.any(result.input_precision_loss)

    @pytest.mark.parametrize("filename", _holdout_failure_files())
    def test_canonical_classification_matches_f64_on_canonical_inputs(self, filename):
        with np.load(HOLDOUT_FAILURE_DIR / filename) as data:
            vertices = data["vertices"].astype(np.float64)
            point, normal = _point_and_normal(data)

        cv, cp, cn = canonicalize_inputs_f32(vertices, point, normal)
        ref = classify_plane_f64(cv, cp, cn)
        cand = classify_plane_f32(vertices, point, normal, POLICY_0_3_0)
        diff = diff_classifications(ref, cand)
        assert diff.agrees, diff.first_divergence


class TestGenuineArithmeticDisagreement:
    """Frozen adversarial case: grid-only quantization produces a genuine
    arithmetic mismatch; Policy 0.3.0's ambiguity fallback corrects it."""

    _GRID_MISMATCH_VERTICES = np.array(
        [
            [9.614375114440918, 17.33206558227539, -28.09351348876953],
            [14.545977592468262, 9.981534957885742, -2.957688331604004],
            [11.79649829864502, -0.8349803686141968, 15.971166610717773],
            [11.393048286437988, 4.124624729156494, -6.225250720977783],
            [0.8737432956695557, -8.755125999450684, -7.019906044006348],
            [1.7978929281234741, -1.2809958457946777, -6.6724090576171875],
            [16.4908504486084, -5.532711505889893, -0.29043489694595337],
        ],
        dtype=np.float64,
    )
    _GRID_MISMATCH_POINT = np.array(
        [1.7978872060775757, -1.2809884548187256, -6.672430515289307],
        dtype=np.float64,
    )
    _GRID_MISMATCH_NORMAL = np.array(
        [-0.000738786009605974, -0.6419697403907776, 0.7667295932769775],
        dtype=np.float64,
    )
    _GRID_ONLY_POLICY = QuantizationPolicy(grid_bits=20, ambiguity_fallback=False)

    def test_inputs_are_f32_representable(self):
        for arr in (
            self._GRID_MISMATCH_VERTICES,
            self._GRID_MISMATCH_POINT,
            self._GRID_MISMATCH_NORMAL,
        ):
            np.testing.assert_array_equal(
                arr, arr.astype(np.float32).astype(np.float64)
            )

    def test_grid_only_produces_arithmetic_mismatch(self):
        case = PlaneCase(
            self._GRID_MISMATCH_VERTICES,
            self._GRID_MISMATCH_POINT,
            self._GRID_MISMATCH_NORMAL,
        )
        result = evaluate_case(case, self._GRID_ONLY_POLICY)
        assert np.any(result.f32_arithmetic_mismatch), (
            "frozen case must produce grid-only mismatch"
        )

    def test_policy_0_3_0_fallback_corrects_mismatch(self):
        case = PlaneCase(
            self._GRID_MISMATCH_VERTICES,
            self._GRID_MISMATCH_POINT,
            self._GRID_MISMATCH_NORMAL,
        )
        result = evaluate_case(case, POLICY_0_3_0)
        assert not np.any(result.f32_arithmetic_mismatch), (
            "ambiguity fallback should correct it"
        )


class TestOnPlaneExcuse:
    """On-plane vertex excused by compare_oracle distance guard (#123)."""

    def test_on_plane_vertex_excused_through_oracle(self):
        vertices = np.array(
            [
                [0.0, 0.0, float(np.float32(1e-7))],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
        oracle_sides = np.array([0, 1, -1], dtype=np.int8)
        plane = TracedPlane(a=0.0, b=0.0, c=1.0, d=0.0, method="test", index=0)
        dummy_f = np.zeros((0, 3), dtype=np.int32)
        clip = TracedClip(
            component_id=0,
            plane=plane,
            pos_verts=0,
            pos_faces=0,
            neg_verts=0,
            neg_faces=0,
            intersection_count=0,
            input_vertices=vertices,
            input_faces=dummy_f,
            oracle_sides=oracle_sides,
        )
        result = compare_oracle(clip, 0, POLICY_0_3_0)
        assert result is not None
        assert result.on_plane_excused == 1
        assert result.num_agree == 2
        assert result.num_disagree == 0


class TestAggregation:
    @pytest.mark.parametrize("filename", _holdout_failure_files())
    def test_raw_shows_mismatch_canonical_shows_none(self, filename):
        with np.load(HOLDOUT_FAILURE_DIR / filename) as data:
            vertices = data["vertices"].astype(np.float64)
            point, normal = _point_and_normal(data)
        case = PlaneCase(vertices, point, normal, label=filename)
        raw = evaluate_case(case)
        canonical = evaluate_case_canonical(case)
        assert raw.num_f32_exact_mismatch > 0
        assert canonical.num_f32_exact_mismatch == 0


class TestCanonicalReplaySelection:
    """End-to-end: --policy 0.3.0 selects canonical replay, 0.2.0 raw."""

    def _parse_args(self, *args):
        with patch("sys.argv", ["evaluate_holdout.py", *args]):
            return evaluate_holdout._parse_args()

    def test_0_3_0_selects_canonical_policy(self):
        _args, policy = self._parse_args(
            "--policy", "0.3.0", "--corpus-manifest", "manifest.json"
        )
        assert policy.canonical_f32_inputs

    def test_0_2_0_retains_raw_policy(self):
        _args, policy = self._parse_args(
            "--policy", "0.2.0", "--corpus-manifest", "manifest.json"
        )
        assert not policy.canonical_f32_inputs

    def test_no_flag_selects_default_policy(self):
        """No --policy flag selects Policy 0.3.0 with canonical inputs."""
        _args, policy = self._parse_args("--corpus-manifest", "manifest.json")
        assert policy.canonical_f32_inputs

    def test_canonical_replay_resolves_precision_loss(self):
        """Diagnostic clip disagrees under raw replay, agrees under canonical."""
        filename = _holdout_failure_files()[0]
        with np.load(HOLDOUT_FAILURE_DIR / filename) as data:
            vertices = data["vertices"].astype(np.float64)
            raw_normal = data["plane_normal"].astype(np.float64)
            offset = float(data["plane_offset"])

        norm_len = float(np.linalg.norm(raw_normal))
        unit_normal = raw_normal / norm_len
        plane = TracedPlane(
            a=float(unit_normal[0]),
            b=float(unit_normal[1]),
            c=float(unit_normal[2]),
            d=offset,
            method="test",
            index=0,
        )
        dummy_v = np.zeros((1, 3), dtype=np.float64)
        dummy_f = np.zeros((1, 3), dtype=np.int32)
        clip = TracedClip(
            component_id=0,
            plane=plane,
            pos_verts=0,
            pos_faces=0,
            neg_verts=0,
            neg_faces=0,
            intersection_count=0,
            pos_vertices=dummy_v,
            pos_triangles=dummy_f,
            neg_vertices=dummy_v,
            neg_triangles=dummy_f,
            input_vertices=vertices,
            input_faces=dummy_f,
        )
        trace = CoACDTrace(
            call_id=0,
            input_vertices=vertices,
            input_faces=dummy_f,
            clips=[clip],
        )

        raw_report = replay_classifications(trace, POLICY_0_2_0)
        canonical_report = replay_classifications(trace, POLICY_0_3_0)

        assert raw_report.num_classification_agree == 0, (
            "diagnostic clip should disagree under raw replay"
        )
        assert canonical_report.num_classification_agree == 1, (
            "should agree under canonical replay"
        )


class TestPolicyConfiguration:
    def test_policy_constants(self):
        assert DEFAULT_POLICY.version == "0.3.0"
        assert DEFAULT_POLICY.canonical_f32_inputs
        assert POLICY_0_2_0.version == "0.2.0"
        assert not POLICY_0_2_0.canonical_f32_inputs
        assert POLICY_0_3_0.version == "0.3.0"
        assert POLICY_0_3_0.canonical_f32_inputs

    def test_policy_0_2_0_mechanical_params_unchanged(self):
        assert POLICY_0_3_0.grid_bits == POLICY_0_2_0.grid_bits
        assert (
            POLICY_0_3_0.classification_ulp_margin
            == POLICY_0_2_0.classification_ulp_margin
        )
        assert (
            POLICY_0_3_0.intersection_snap_bits == POLICY_0_2_0.intersection_snap_bits
        )
        assert POLICY_0_3_0.ambiguity_fallback == POLICY_0_2_0.ambiguity_fallback
        assert POLICY_0_3_0.winding_check == POLICY_0_2_0.winding_check

    def test_default_policy_is_0_3_0(self):
        assert DEFAULT_POLICY.version == "0.3.0"
        assert DEFAULT_POLICY.grid_bits == 20
        assert DEFAULT_POLICY.ambiguity_fallback
        assert DEFAULT_POLICY.canonical_f32_inputs


class TestPredicateCanonicalContract:
    """Defect 1: canonical f32 inputs materially change the grid path."""

    def test_classify_input_invariant_under_canonicalization(self):
        """classify_plane_f32(source) == classify_plane_f32(canonical) under POLICY_0_3_0."""
        filename = _holdout_failure_files()[0]
        with np.load(HOLDOUT_FAILURE_DIR / filename) as data:
            vertices = data["vertices"].astype(np.float64)
            point, normal = _point_and_normal(data)

        cv, cp, cn = canonicalize_inputs_f32(vertices, point, normal)
        assert not np.array_equal(vertices, cv), "need non-f32-representable inputs"

        src_result = classify_plane_f32(vertices, point, normal, POLICY_0_3_0)
        canon_result = classify_plane_f32(cv, cp, cn, POLICY_0_3_0)
        np.testing.assert_array_equal(src_result.signs, canon_result.signs)

    def test_canonicalization_changes_grid_path(self):
        """Large-offset cancellation: f32 collapses sub-ULP differences."""
        v = np.array(
            [[1e8, 0, 0], [1e8 + 1, 0, 0], [1e8 + 2, 0, 0]],
            dtype=np.float64,
        )
        p = np.array([1e8 + 0.5, 0, 0], dtype=np.float64)
        n = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        no_canon = QuantizationPolicy(
            grid_bits=20, ambiguity_fallback=False, canonical_f32_inputs=False
        )
        with_canon = QuantizationPolicy(
            grid_bits=20, ambiguity_fallback=False, canonical_f32_inputs=True
        )

        raw = classify_plane_f32(v, p, n, no_canon)
        canon = classify_plane_f32(v, p, n, with_canon)

        np.testing.assert_array_equal(raw.signs, np.array([-1, 1, 1]))
        np.testing.assert_array_equal(canon.signs, np.array([0, 0, 0]))


class TestMixedClipPrecisionLoss:
    """Defect 2: per-vertex analysis finds both precision-loss and genuine-arithmetic
    in one clip. Set arithmetic at clip level would miss the precision-loss."""

    _PLANE_POINT = np.array(
        [1.7978872060775757, -1.2809884548187256, -6.672430515289307],
        dtype=np.float64,
    )
    _PLANE_NORMAL = np.array(
        [-0.000738786009605974, -0.6419697403907776, 0.7667295932769775],
        dtype=np.float64,
    )
    _MIXED_VERTICES = np.array(
        [
            [9.614375114440918, 17.33206558227539, -28.09351348876953],
            [14.545977592468262, 9.981534957885742, -2.957688331604004],
            [1.7978929281234741, -1.2809958457946777, -6.6724090576171875],
            [3.086140502712534, -4.554606557523074, -9.412134496544875],
        ],
        dtype=np.float64,
    )
    _GRID_ONLY_POLICY = QuantizationPolicy(
        grid_bits=20, ambiguity_fallback=False, canonical_f32_inputs=False
    )

    def test_mixed_clip_has_both_categories(self):
        """categorize_disagreements finds both categories in this clip."""
        v, p, n = self._MIXED_VERTICES, self._PLANE_POINT, self._PLANE_NORMAL
        policy = self._GRID_ONLY_POLICY

        raw_ref = classify_plane_f64(v, p, n)
        cv, cp, cn = canonicalize_inputs_f32(v, p, n)
        canon_ref = classify_plane_f64(cv, cp, cn)
        canon_cand = classify_plane_f32(cv, cp, cn, policy)

        prec_loss, genuine = categorize_disagreements(
            raw_ref.signs, canon_ref.signs, canon_cand.signs
        )

        assert np.any(prec_loss), "should have precision-loss vertex"
        assert np.any(genuine), "should have genuine-arithmetic vertex"

    def test_precision_loss_is_reference_only(self):
        """Precision loss = raw_ref != canon_ref, independent of candidate."""
        v, p, n = self._MIXED_VERTICES, self._PLANE_POINT, self._PLANE_NORMAL

        raw_ref = classify_plane_f64(v, p, n)
        cv, cp, cn = canonicalize_inputs_f32(v, p, n)
        canon_ref = classify_plane_f64(cv, cp, cn)

        prec_loss = raw_ref.signs != canon_ref.signs
        assert prec_loss[3] and not any(prec_loss[:3])
