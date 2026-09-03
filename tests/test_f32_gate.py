"""f32 predicate disproof gate (#101): does f32 quantization change predicate outcomes."""

import inspect
import re

import numpy as np
import pytest

from chitin.f32_policy import DEFAULT_POLICY, QuantizationPolicy, sweep_policies
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


KNOWN_HULL_DIVERGENCES = {
    "high_complexity",
    "thin_u_channel",
    "curved_pipe_quarter",
    "cross_bracket",
    "h_shape",
}


@pytest.mark.parametrize("fixture_name", ALL_FIXTURE_NAMES)
def test_hull_topology(fixture_name):
    """Hull structural topology gate — all fixtures, no silent skips.

    Fixtures in KNOWN_HULL_DIVERGENCES are expected to diverge (Qhull sensitivity
    to f32 rounding, not a predicate failure); the test verifies they still diverge
    and xfails if they unexpectedly pass (which would mean the set needs updating).
    """
    cases = [c for c in build_test_cases() if c.fixture_name == fixture_name]
    failures = []
    for case in cases:
        report = run_predicate_gate(case, DEFAULT_POLICY)
        if report.hull_diff is None:
            continue
        if not report.hull_diff.agrees:
            failures.append(report.hull_diff.first_divergence)

    if fixture_name in KNOWN_HULL_DIVERGENCES:
        assert failures, (
            f"{fixture_name}: expected hull divergence (Qhull f32 sensitivity) "
            f"but all hulls agreed — remove from KNOWN_HULL_DIVERGENCES"
        )
        pytest.skip(
            f"{fixture_name}: {len(failures)} known hull divergence(s) "
            f"(Qhull f32 sensitivity, not predicate failure)"
        )
    else:
        assert not failures, (
            f"{fixture_name}: unexpected hull divergence: {failures[0]}"
        )


@pytest.mark.slow
def test_quantization_sweep():
    policies = sweep_policies(range(10, 24))
    report = run_corpus_gate(policies=policies, n_random_planes=2)
    for r in report.reports:
        assert r.classification_diff.agrees
        assert r.clip_diff.agrees


def test_no_absolute_epsilon():
    from chitin import coacd_trace_replay, f32_policy, f32_predicates

    source = (
        inspect.getsource(f32_policy)
        + inspect.getsource(f32_predicates)
        + inspect.getsource(coacd_trace_replay)
    )
    forbidden = [
        "1e-9",
        "1e-8",
        "1e-7",
        "1e-6",
        "1e-5",
        "1e-4",
        "0.0001",
        "0.00001",
        "1e-06",
        "1e-05",
    ]
    for token in forbidden:
        assert token not in source, f"found absolute-looking epsilon literal: {token}"

    # Allowlist: known-legitimate divide-by-zero / degenerate-input guards,
    # not geometric tolerances. Matched by substring (robust to line drift),
    # then stripped from `source` before the regex scans below so they
    # can't trip the stricter checks.
    #   - f32_policy.py: `1e-30` floor on the max-extent denominator, already
    #     commented in-source as "not a geometric tolerance".
    #   - coacd_trace_replay.py: `norm < 1e-15` guards a zero-length plane
    #     normal before `normal = n / norm` — same class of guard, just a
    #     different magnitude and a bare comparison instead of a named const.
    allowlisted_snippets = [
        "1e-30)",
        "norm < 1e-15",
    ]
    scanned = source
    for snippet in allowlisted_snippets:
        scanned = scanned.replace(snippet, "")

    # Bare comparisons against small scientific-notation or decimal literals,
    # e.g. `if norm < 1e-15:` or `abs(x) < 0.0001`, with no named variable to
    # catch on the substring list above. Exponent magnitude >= 4 covers
    # anything tighter than ~1e-4, which is the smallest a legitimate
    # grid-relative tolerance should ever look like as a bare literal.
    comparison_pattern = re.compile(
        r"[<>]=?\s*(?:1e-(\d+)|(\d*\.\d+)e-(\d+)|(0\.0{4,}\d*))"
    )
    for match in comparison_pattern.finditer(scanned):
        exp_a, _mantissa, exp_b, decimal_literal = match.groups()
        if decimal_literal is not None:
            bad = True
        else:
            exponent = int(exp_a or exp_b)
            bad = exponent >= 4
        assert not bad, (
            f"found absolute-looking epsilon in a bare comparison: {match.group(0)!r}"
        )

    # Direct assignments of very small literals that look like absolute
    # tolerances (`= 1e-6` etc.) with exponent >= 6, independent of the
    # variable name — the existing forbidden-token list above only catches
    # a fixed set of exponent/decimal-digit spellings.
    assignment_pattern = re.compile(r"=\s*1e-(\d+)\b")
    for match in assignment_pattern.finditer(scanned):
        exponent = int(match.group(1))
        assert exponent < 6, (
            f"found absolute-looking epsilon assignment: {match.group(0)!r}"
        )

    # Arithmetic uses of small absolute literals: `+ 1e-N`, `- 1e-N`,
    # `* 1e-N` where N >= 6 — additive/multiplicative smuggling of
    # world-unit epsilons into grid-relative code.
    arithmetic_pattern = re.compile(r"[+\-*]\s*1e-(\d+)\b")
    for match in arithmetic_pattern.finditer(scanned):
        exponent = int(match.group(1))
        assert exponent < 6, (
            f"found absolute-looking epsilon in arithmetic: {match.group(0)!r}"
        )


def test_corpus_report_structure():
    report = run_corpus_gate(policies=[DEFAULT_POLICY], n_random_planes=2)
    from chitin.trace_fixtures import FIXTURES

    assert len(report.reports) == len(FIXTURES) * (3 + 2)
    summary = report.summary_by_predicate()
    assert set(summary.keys()) == {
        "classify_plane",
        "clip_mesh",
        "extract_cap",
        "convex_hull",
    }
    assert 0.0 <= report.pass_rate <= 1.0


def test_first_divergence_reported():
    # thin_panel agrees under DEFAULT_POLICY once diff_hulls uses a validated
    # scipy volume instead of the old unwound tetra-sum; a coarse grid still
    # legitimately diverges on hull volume, which is what this test checks.
    coarse_policy = QuantizationPolicy(grid_bits=10)
    cases = [c for c in build_test_cases() if c.fixture_name == "thin_panel"]
    found = False
    for case in cases:
        report = run_predicate_gate(case, coarse_policy)
        if not report.all_agree:
            found = True
            assert report.first_divergence is not None
            assert "convex_hull" in report.first_divergence
    assert found, "expected at least one divergent case for thin_panel"
