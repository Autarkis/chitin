"""Tests for boundary-directed exact and metamorphic predicate search."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from chitin.f32_adversarial import (
    PlaneCase,
    classify_plane_exact,
    evaluate_case,
    generate_boundary_case,
    permute_vertices,
    power_of_two_scale,
    search_adversaries,
    shrink_f32_exact_failure,
    translate,
    write_findings,
)

HOLDOUT_FAILURE_DIR = (
    Path(__file__).parent / "fixtures" / "traces" / "holdout_failures_0_2_0"
)


def _holdout_failure_files() -> list[str]:
    manifest = json.loads((HOLDOUT_FAILURE_DIR / "manifest.json").read_text())
    return [entry["file"] for entry in manifest["clips"]]


def _case() -> PlaneCase:
    return PlaneCase(
        vertices=np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        plane_point=np.zeros(3, dtype=np.float64),
        plane_normal=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        label="simple",
    )


class TestExactInputOracle:
    def test_classifies_finite_binary_inputs_exactly(self):
        case = _case()
        np.testing.assert_array_equal(
            classify_plane_exact(case), np.array([-1, 0, 1, 0], dtype=np.int8)
        )

    def test_resolves_subnormal_distance(self):
        tiny = np.nextafter(0.0, 1.0)
        case = PlaneCase(
            vertices=np.array([[tiny, 0.0, 0.0]], dtype=np.float64),
            plane_point=np.zeros(3, dtype=np.float64),
            plane_normal=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )
        assert classify_plane_exact(case).tolist() == [1]

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_rejects_nonfinite_inputs(self, bad):
        case = _case()
        vertices = case.vertices.copy()
        vertices[0, 0] = bad
        with pytest.raises(ValueError, match="finite"):
            classify_plane_exact(
                PlaneCase(vertices, case.plane_point, case.plane_normal)
            )

    @pytest.mark.parametrize("filename", _holdout_failure_files())
    def test_confirms_policy_0_2_holdout_reference(self, filename):
        with np.load(HOLDOUT_FAILURE_DIR / filename) as data:
            vertices = data["vertices"].astype(np.float64)
            normal = data["plane_normal"].astype(np.float64)
            normal /= np.linalg.norm(normal)
            offset = float(data["plane_offset"])
        point = normal * (-offset / np.dot(normal, normal))
        case = PlaneCase(vertices, point, normal, label=filename)

        result = evaluate_case(case)
        np.testing.assert_array_equal(result.exact_signs, result.f64_signs)
        assert result.num_f32_exact_mismatch > 0


class TestBoundaryGenerator:
    def test_is_seed_deterministic(self):
        first = generate_boundary_case(np.random.default_rng(41), 3)
        second = generate_boundary_case(np.random.default_rng(41), 3)
        np.testing.assert_array_equal(first.vertices, second.vertices)
        np.testing.assert_array_equal(first.plane_point, second.plane_point)
        np.testing.assert_array_equal(first.plane_normal, second.plane_normal)

    def test_targets_a_real_policy_boundary(self):
        case = generate_boundary_case(np.random.default_rng(7), 0)
        result = evaluate_case(case)
        assert result.num_f32_exact_mismatch > 0
        assert result.num_f64_exact_mismatch == 0


class TestMetamorphicTransforms:
    def test_vertex_permutation_preserves_classification(self):
        case = _case()
        order = np.array([2, 0, 3, 1])
        baseline = evaluate_case(case)
        transformed = evaluate_case(permute_vertices(case, order))
        inverse = np.argsort(order)
        np.testing.assert_array_equal(
            transformed.exact_signs[inverse], baseline.exact_signs
        )
        np.testing.assert_array_equal(
            transformed.f32_signs[inverse], baseline.f32_signs
        )

    @pytest.mark.parametrize("exponent", [-8, -1, 0, 7])
    def test_power_of_two_scale_preserves_exact_sign(self, exponent):
        baseline = classify_plane_exact(_case())
        transformed = classify_plane_exact(power_of_two_scale(_case(), exponent))
        np.testing.assert_array_equal(transformed, baseline)

    def test_translation_preserves_exact_sign_when_representable(self):
        baseline = classify_plane_exact(_case())
        transformed = classify_plane_exact(
            translate(_case(), np.array([8.0, -4.0, 2.0]))
        )
        np.testing.assert_array_equal(transformed, baseline)


class TestSearchAndReduction:
    def test_search_is_deterministic_and_finds_boundaries(self):
        first = search_adversaries(seed=19, cases=8)
        second = search_adversaries(seed=19, cases=8)
        assert len(first) == len(second) > 0
        assert [f.case.label for f in first] == [f.case.label for f in second]
        assert [f.failure_types for f in first] == [f.failure_types for f in second]
        assert [f.coverage_bucket for f in first] == [f.coverage_bucket for f in second]
        assert len({f.coverage_bucket for f in first}) == len(first)
        np.testing.assert_allclose(
            [f.score for f in first], [f.score for f in second], rtol=0, atol=0
        )

    def test_shrinker_preserves_failure(self):
        case = generate_boundary_case(np.random.default_rng(23), 0)
        assert len(case.vertices) > 1
        shrunk = shrink_f32_exact_failure(case)
        assert len(shrunk.vertices) < len(case.vertices)
        assert evaluate_case(shrunk).num_f32_exact_mismatch > 0

    def test_writes_digest_bearing_immutable_reproducers(self, tmp_path):
        findings = search_adversaries(seed=5, cases=2)
        output = tmp_path / "findings"
        manifest_path = write_findings(findings, output, seed=5)
        manifest = json.loads(manifest_path.read_text())

        assert manifest["seed"] == 5
        assert len(manifest["findings"]) == len(findings)
        for entry in manifest["findings"]:
            payload = (output / entry["file"]).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
            assert entry["vertices_after"] <= entry["vertices_before"]

        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            write_findings(findings, output, seed=5)
