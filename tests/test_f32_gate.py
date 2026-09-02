"""f32 predicate disproof gate (#101): does f32 quantization change predicate outcomes."""

import inspect

import numpy as np
import pytest

from chitin.f32_policy import DEFAULT_POLICY, sweep_policies
from chitin.f32_predicates import (
    classify_plane_f32,
    classify_plane_f64,
    clip_mesh_f32,
    clip_mesh_f64,
    diff_caps,
    diff_classifications,
    diff_clips,
    extract_cap_f32,
    extract_cap_f64,
)
from chitin.f32_replay import (
    build_test_cases,
    generate_test_planes,
    run_corpus_gate,
    run_predicate_gate,
)
from chitin.trace_fixtures import FIXTURES

ALL_FIXTURE_NAMES = list(FIXTURES.keys())
ORDINARY_FIXTURE_NAMES = ["box", "l_shape", "disconnected"]
ADVERSARIAL_FIXTURE_NAMES = ["thin_panel", "degenerate", "high_complexity"]


@pytest.mark.parametrize("fixture_name", ALL_FIXTURE_NAMES)
def test_classification_agreement(fixture_name):
    vertices, _faces = FIXTURES[fixture_name]()
    vertices = vertices.astype(np.float64)
    for plane_point, plane_normal in generate_test_planes(
        vertices, seed=42, n_random=5
    ):
        ref = classify_plane_f64(vertices, plane_point, plane_normal)
        cand = classify_plane_f32(vertices, plane_point, plane_normal, DEFAULT_POLICY)
        diff = diff_classifications(ref, cand)
        assert diff.agrees, diff.first_divergence
        assert np.array_equal(ref.signs, cand.signs)


@pytest.mark.parametrize("fixture_name", ALL_FIXTURE_NAMES)
def test_clip_agreement(fixture_name):
    vertices, faces = FIXTURES[fixture_name]()
    vertices = vertices.astype(np.float64)
    faces = faces.astype(np.int64)
    for plane_point, plane_normal in generate_test_planes(
        vertices, seed=42, n_random=5
    ):
        ref = clip_mesh_f64(vertices, faces, plane_point, plane_normal)
        cand = clip_mesh_f32(vertices, faces, plane_point, plane_normal, DEFAULT_POLICY)
        diff = diff_clips(ref, cand)
        assert diff.agrees, diff.first_divergence


@pytest.mark.parametrize("fixture_name", ALL_FIXTURE_NAMES)
def test_cap_agreement(fixture_name):
    vertices, faces = FIXTURES[fixture_name]()
    vertices = vertices.astype(np.float64)
    faces = faces.astype(np.int64)
    for plane_point, plane_normal in generate_test_planes(
        vertices, seed=42, n_random=5
    ):
        clip_ref = clip_mesh_f64(vertices, faces, plane_point, plane_normal)
        clip_cand = clip_mesh_f32(
            vertices, faces, plane_point, plane_normal, DEFAULT_POLICY
        )
        cap_ref = extract_cap_f64(clip_ref)
        cap_cand = extract_cap_f32(clip_cand, DEFAULT_POLICY)
        diff = diff_caps(cap_ref, cap_cand)
        assert diff.agrees, diff.first_divergence


@pytest.mark.parametrize("fixture_name", ORDINARY_FIXTURE_NAMES)
def test_hull_topology_ordinary(fixture_name):
    cases = [c for c in build_test_cases() if c.fixture_name == fixture_name]
    for case in cases:
        report = run_predicate_gate(case, DEFAULT_POLICY)
        if report.hull_diff is None:
            continue
        assert (
            report.hull_diff.details["ref_face_count"]
            == report.hull_diff.details["cand_face_count"]
        )
        assert (
            report.hull_diff.details["ref_outward_consistent"]
            == report.hull_diff.details["cand_outward_consistent"]
        )


@pytest.mark.parametrize("fixture_name", ADVERSARIAL_FIXTURE_NAMES)
def test_hull_adversarial(fixture_name):
    cases = [c for c in build_test_cases() if c.fixture_name == fixture_name]
    for case in cases:
        report = run_predicate_gate(case, DEFAULT_POLICY)
        assert report.classification_diff.agrees
        assert report.clip_diff.agrees
        assert report.cap_diff.agrees


@pytest.mark.slow
def test_quantization_sweep():
    policies = sweep_policies(range(10, 24))
    report = run_corpus_gate(policies=policies, n_random_planes=2)
    for r in report.reports:
        assert r.classification_diff.agrees
        assert r.clip_diff.agrees


def test_no_absolute_epsilon():
    from chitin import f32_policy, f32_predicates

    source = inspect.getsource(f32_policy) + inspect.getsource(f32_predicates)
    forbidden = ["1e-6", "1e-5", "1e-4", "0.0001", "0.00001", "1e-06", "1e-05"]
    for token in forbidden:
        assert token not in source, f"found absolute-looking epsilon literal: {token}"


def test_corpus_report_structure():
    report = run_corpus_gate(policies=[DEFAULT_POLICY], n_random_planes=2)
    assert len(report.reports) == 6 * (3 + 2)
    summary = report.summary_by_predicate()
    assert set(summary.keys()) == {
        "classify_plane",
        "clip_mesh",
        "extract_cap",
        "convex_hull",
    }
    assert 0.0 <= report.pass_rate <= 1.0


def test_first_divergence_reported():
    cases = [c for c in build_test_cases() if c.fixture_name == "thin_panel"]
    found = False
    for case in cases:
        report = run_predicate_gate(case, DEFAULT_POLICY)
        if not report.all_agree:
            found = True
            assert report.first_divergence is not None
            assert "convex_hull" in report.first_divergence
    assert found, "expected at least one divergent case for thin_panel"
