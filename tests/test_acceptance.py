import pytest

from chitin._metric_names import (
    CLEARANCE_BLOCKED_FRACTION,
    COLLIDER_VOLUME_PRECISION,
    DEEP_FALSE_FILL_FRACTION,
    FALLBACK_RATIO,
    FALSE_FILL_FRACTION,
    HULL_COUNT,
    HULL_TRIANGLE_COUNT,
    HULL_VERTEX_COUNT,
    PLANAR_SUBSTITUTE_HULLS,
    PROBE_COVERAGE,
    PROBE_GAP_CLUSTERS,
    RADIUS_BLOCKED_FRACTION,
    SEAM_SNAG_COUNT,
    SNUG_FIT_STATUS,
    SOURCE_SURFACE_COVERAGE,
    STANDABLE_FRACTION,
    SWEEP_TRAVERSABILITY,
    WORST_COMPONENT_SURFACE_COVERAGE,
)
from chitin.acceptance import (
    PROFILES,
    AcceptancePolicy,
    apply_profile,
    evaluate,
    get_profile,
    record_artifact_checks,
    report_metrics,
)
from chitin.config import Config

# A clean, fully-covered single-unit build (no per-cell split, no fallback).
CLEAN = {
    HULL_COUNT: 6,
    SOURCE_SURFACE_COVERAGE: 0.99,
    WORST_COMPONENT_SURFACE_COVERAGE: None,
    FALSE_FILL_FRACTION: 0.05,
    DEEP_FALSE_FILL_FRACTION: 0.02,
    COLLIDER_VOLUME_PRECISION: 0.95,
    "fallback_hulls": 0,
    FALLBACK_RATIO: 0.0,
    PLANAR_SUBSTITUTE_HULLS: 0,
    "coacd_deterministic": True,
    SNUG_FIT_STATUS: "applied",
    PROBE_COVERAGE: 0.95,
    PROBE_GAP_CLUSTERS: 0,
    SWEEP_TRAVERSABILITY: 0.95,
    STANDABLE_FRACTION: 0.95,
    CLEARANCE_BLOCKED_FRACTION: 0.05,
    RADIUS_BLOCKED_FRACTION: 0.05,
    SEAM_SNAG_COUNT: 0,
    HULL_VERTEX_COUNT: 1000,
    HULL_TRIANGLE_COUNT: 2000,
    "compile_ms": None,
}


def _with(**over):
    return {**CLEAN, **over}


# --- evaluate: the pure verdict function --------------------------------


def test_interactive_passes_anything():
    # Permissive: no checks, so even an empty/degenerate report is accepted.
    policy = get_profile("interactive").policy
    assert evaluate(policy, {}).passed
    assert evaluate(policy, _with(**{HULL_COUNT: 0}, fallback_hulls=3)).passed


def test_robotics_rejects_fallback_hulls():
    policy = get_profile("robotics").policy
    verdict = evaluate(policy, _with(fallback_hulls=1))
    assert not verdict.passed
    assert any("fallback" in r for r in verdict.reasons)


def test_robotics_rejects_nondeterministic_build():
    # --fast lets CoACD's threads pick a different decomposition each run, so
    # the shipped hulls cannot be reproduced from the manifest that describes
    # them. A collider a simulation is validated against has to be.
    policy = get_profile("robotics").policy
    verdict = evaluate(policy, _with(coacd_deterministic=False))
    assert not verdict.passed
    assert any("reproduce" in r for r in verdict.reasons)


def test_robotics_rejects_skipped_snug_fit():
    policy = get_profile("robotics").policy
    verdict = evaluate(policy, _with(**{SNUG_FIT_STATUS: "skipped"}))
    assert not verdict.passed
    check = next(check for check in verdict.checks if check.name == "snug_fit_applied")
    assert check.suggestion


def test_robotics_accepts_a_build_with_no_coacd_run():
    # Absent flag means no decomposition ran at all (planar box, environment
    # shell): there is no unreproducible search to reject.
    verdict = evaluate(get_profile("robotics").policy, _with(coacd_deterministic=None))
    assert verdict.passed


def test_walkable_allows_nondeterministic_build():
    # Reproducibility is gated for robotics only; a walkable floor plate is
    # allowed to trade it for speed.
    assert evaluate(
        get_profile("walkable").policy, _with(coacd_deterministic=False)
    ).passed


def test_robotics_passes_clean_build():
    verdict = evaluate(get_profile("robotics").policy, CLEAN)
    assert verdict.passed
    assert verdict.reasons == []


def test_robotics_requires_hulls():
    verdict = evaluate(get_profile("robotics").policy, _with(**{HULL_COUNT: 0}))
    assert not verdict.passed
    assert any("no hulls" in r for r in verdict.reasons)


def test_walkable_coverage_gate():
    policy = get_profile("walkable").policy
    assert evaluate(policy, _with(**{SOURCE_SURFACE_COVERAGE: 0.90})).passed
    low = evaluate(policy, _with(**{SOURCE_SURFACE_COVERAGE: 0.50}))
    assert not low.passed
    assert any(SOURCE_SURFACE_COVERAGE in r for r in low.reasons)


def test_walkable_bounds_fallback_ratio():
    policy = get_profile("walkable").policy
    assert evaluate(policy, _with(fallback_hulls=1, **{FALLBACK_RATIO: 1 / 6})).passed
    verdict = evaluate(policy, _with(fallback_hulls=2, **{FALLBACK_RATIO: 2 / 6}))
    assert not verdict.passed
    assert any(check.name == "fallback_ratio" for check in verdict.checks)


def test_worst_cell_gate_absent_passes_but_low_fails():
    policy = AcceptancePolicy("t", mode="strict", min_worst_cell_fraction=0.7)
    assert evaluate(policy, _with(**{WORST_COMPONENT_SURFACE_COVERAGE: None})).passed
    assert evaluate(policy, _with(**{WORST_COMPONENT_SURFACE_COVERAGE: 0.9})).passed
    assert not evaluate(policy, _with(**{WORST_COMPONENT_SURFACE_COVERAGE: 0.3})).passed


def test_missing_coverage_metric_fails_gate():
    # A coverage threshold with no measured coverage is a failure, not a pass.
    policy = AcceptancePolicy("t", mode="strict", min_covered_fraction=0.9)
    assert not evaluate(policy, _with(**{SOURCE_SURFACE_COVERAGE: None})).passed


def test_verdict_to_dict_shape():
    verdict = evaluate(get_profile("robotics").policy, _with(fallback_hulls=1))
    d = verdict.to_dict()
    assert d["profile"] == "robotics"
    assert d["passed"] is False
    assert isinstance(d["reasons"], list) and d["reasons"]
    assert all({"check", "passed", "detail"} <= c.keys() for c in d["checks"])


# --- volume metrics: false fill -----------------------------------------


def test_robotics_rejects_high_false_fill():
    """A robotics collider with >30% phantom volume is rejected."""
    policy = get_profile("robotics").policy
    verdict = evaluate(policy, _with(**{FALSE_FILL_FRACTION: 0.50}))
    assert not verdict.passed
    assert any("false_fill" in c.name for c in verdict.checks if not c.passed)


def test_robotics_passes_low_false_fill():
    policy = get_profile("robotics").policy
    verdict = evaluate(policy, _with(**{FALSE_FILL_FRACTION: 0.15}))
    assert verdict.passed


def test_robotics_rejects_high_deep_false_fill():
    policy = get_profile("robotics").policy
    verdict = evaluate(policy, _with(**{DEEP_FALSE_FILL_FRACTION: 0.35}))
    assert not verdict.passed


def test_walkable_rejects_high_false_fill():
    """Walkable tolerates more but still rejects >50%."""
    policy = get_profile("walkable").policy
    verdict = evaluate(policy, _with(**{FALSE_FILL_FRACTION: 0.60}))
    assert not verdict.passed


def test_walkable_passes_moderate_false_fill():
    policy = get_profile("walkable").policy
    verdict = evaluate(policy, _with(**{FALSE_FILL_FRACTION: 0.40}))
    assert verdict.passed


def test_missing_required_volume_metrics_fail_gate():
    policy = get_profile("robotics").policy
    verdict = evaluate(
        policy, _with(**{FALSE_FILL_FRACTION: None, DEEP_FALSE_FILL_FRACTION: None})
    )
    assert not verdict.passed
    failed = [check for check in verdict.checks if not check.passed]
    assert {check.name for check in failed} >= {"false_fill", "deep_false_fill"}
    assert all(check.suggestion for check in failed)


# --- walkable probe -------------------------------------------------------


def test_walkable_probe_rejects_low_coverage():
    """An obstructed walkable fixture with low probe coverage fails."""
    policy = get_profile("walkable").policy
    verdict = evaluate(policy, _with(**{PROBE_COVERAGE: 0.40, PROBE_GAP_CLUSTERS: 2}))
    assert not verdict.passed
    assert any("probe_coverage" in c.name for c in verdict.checks if not c.passed)


def test_walkable_probe_rejects_many_gaps():
    policy = get_profile("walkable").policy
    verdict = evaluate(policy, _with(**{PROBE_COVERAGE: 0.90, PROBE_GAP_CLUSTERS: 10}))
    assert not verdict.passed
    assert any("probe_gap_clusters" in c.name for c in verdict.checks if not c.passed)


def test_walkable_probe_passes_good_floor():
    policy = get_profile("walkable").policy
    verdict = evaluate(policy, _with(**{PROBE_COVERAGE: 0.85, PROBE_GAP_CLUSTERS: 2}))
    assert verdict.passed


def test_walkable_probe_absent_fails():
    policy = get_profile("walkable").policy
    verdict = evaluate(
        policy, _with(**{PROBE_COVERAGE: None, PROBE_GAP_CLUSTERS: None})
    )
    assert not verdict.passed
    assert all(check.suggestion for check in verdict.checks if not check.passed)


# --- walkable capsule sweep -----------------------------------------------


def test_walkable_sweep_rejects_disconnected_or_blocked_floor():
    policy = get_profile("walkable").policy
    verdict = evaluate(
        policy,
        _with(
            **{
                SWEEP_TRAVERSABILITY: 0.45,
                STANDABLE_FRACTION: 0.60,
                CLEARANCE_BLOCKED_FRACTION: 0.35,
            }
        ),
    )
    assert not verdict.passed
    failed = {check.name for check in verdict.checks if not check.passed}
    assert failed >= {
        "capsule_traversability",
        "capsule_standable_fraction",
        "capsule_clearance",
    }


def test_walkable_sweep_absent_fails():
    policy = get_profile("walkable").policy
    verdict = evaluate(
        policy,
        _with(
            **{
                SWEEP_TRAVERSABILITY: None,
                STANDABLE_FRACTION: None,
                CLEARANCE_BLOCKED_FRACTION: None,
            }
        ),
    )
    assert not verdict.passed
    assert all(check.suggestion for check in verdict.checks if not check.passed)


def test_obstructed_walkable_fixture_fails_capsule_gate(box_hull):
    from chitin.plan import BuildPlan
    from chitin.result import ExtractionResult

    floor = box_hull(center=(0.0, -0.05, 0.0), half=(2.0, 0.05, 2.0))
    wall = box_hull(center=(0.0, 1.0, 0.0), half=(0.05, 1.0, 2.0))
    plan = BuildPlan(input_kind="mesh")
    plan.detected["coverage"] = {
        SOURCE_SURFACE_COVERAGE: 0.99,
        FALSE_FILL_FRACTION: 0.05,
        DEEP_FALSE_FILL_FRACTION: 0.01,
    }
    result = ExtractionResult(
        hulls=[floor, wall],
        source_vertex_count=16,
        mesh_vertex_count=16,
        build_plan=plan,
    )

    policy = get_profile("walkable").policy
    record_artifact_checks(result, policy)
    metrics = report_metrics(result)
    verdict = evaluate(policy, metrics)

    assert metrics[RADIUS_BLOCKED_FRACTION] > 0
    assert metrics[SWEEP_TRAVERSABILITY] < policy.min_sweep_traversability
    assert not verdict.passed
    assert any(check.name == "capsule_traversability" for check in verdict.checks)


# --- suggestions ------------------------------------------------------------


def test_failed_check_has_suggestion():
    """Every failed check carries an actionable suggestion."""
    policy = get_profile("robotics").policy
    verdict = evaluate(policy, _with(fallback_hulls=3))
    failed = [c for c in verdict.checks if not c.passed]
    assert len(failed) > 0
    for c in failed:
        assert c.suggestion is not None
        assert len(c.suggestion) > 10


def test_passing_check_has_no_suggestion():
    policy = get_profile("robotics").policy
    verdict = evaluate(policy, _with())
    for c in verdict.checks:
        assert c.passed
        assert c.suggestion is None


# --- optional gates: hull vertex count, compile latency ---------------------


def test_hull_vertex_count_gate():
    """Custom policy with max_hull_vertices gates total vertex count."""
    policy = AcceptancePolicy("test", max_hull_vertices=500)
    verdict = evaluate(policy, _with(**{HULL_VERTEX_COUNT: 1000}))
    assert not verdict.passed
    assert any("hull_vertex_count" in c.name for c in verdict.checks if not c.passed)


def test_hull_triangle_count_gate():
    policy = AcceptancePolicy("test", max_hull_triangles=500)
    verdict = evaluate(policy, _with(**{HULL_TRIANGLE_COUNT: 1000}))
    assert not verdict.passed
    assert any("hull_triangle_count" in c.name for c in verdict.checks if not c.passed)


def test_compile_latency_gate():
    policy = AcceptancePolicy("test", max_compile_ms=5000.0)
    verdict = evaluate(policy, _with(compile_ms=8000.0))
    assert not verdict.passed
    verdict_ok = evaluate(policy, _with(compile_ms=3000.0))
    assert verdict_ok.passed


def test_planar_vs_fallback_visible():
    """Planar substitutes and failure fallbacks are independently visible in report_metrics."""
    import numpy as np

    from chitin.plan import BuildPlan
    from chitin.result import ExtractionResult, Hull

    plan = BuildPlan(input_kind="mesh")
    plan.detected["fallback_hulls"] = 1
    plan.detected["planar_substitute_hulls"] = 3
    plan.detected["coverage"] = {}
    plan.detected["coacd_deterministic"] = True

    hull = Hull(
        vertices=np.zeros((4, 3), dtype=np.float32),
        indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.int32),
    )
    result = ExtractionResult(
        hulls=[hull],
        source_vertex_count=100,
        mesh_vertex_count=100,
        build_plan=plan,
    )
    metrics = report_metrics(result)
    assert metrics["fallback_hulls"] == 1
    assert metrics[FALLBACK_RATIO] == 1.0
    assert metrics[PLANAR_SUBSTITUTE_HULLS] == 3


# --- apply_profile: config precedence -----------------------------------


def test_apply_profile_fills_defaults():
    applied = apply_profile(Config(), get_profile("robotics"))
    assert applied.concavity == 0.05  # preset filled the default
    assert applied.snug_fit is True


def test_apply_profile_respects_explicit_values():
    # A caller who set concavity keeps it; the profile only fills untouched fields.
    applied = apply_profile(Config(concavity=0.08), get_profile("robotics"))
    assert applied.concavity == 0.08
    assert applied.snug_fit is True  # still filled where left at default


def test_apply_profile_respects_an_explicit_default_value():
    # Comparing against Config() cannot tell "--concavity 0.05" (the default,
    # deliberately typed) from no flag at all, so the caller passes the set of
    # fields it actually saw and the profile leaves those alone.
    default = Config().concavity
    applied = apply_profile(
        Config(concavity=default), get_profile("robotics"), explicit={"concavity"}
    )
    assert applied.concavity == default
    # Fields the caller did not supply still get the preset.
    assert applied.snug_fit is True


def test_cli_maps_every_preset_field_to_a_flag():
    # cli.py translates preset Config fields to argparse dests to decide what
    # was explicit; a new preset field without a mapping would silently revert
    # to the old guesswork.
    from chitin.cli import PRESET_FLAG_DESTS

    preset_fields = {f for p in PROFILES.values() for f in p.preset}
    assert preset_fields <= set(PRESET_FLAG_DESTS)


def test_interactive_profile_is_a_noop():
    base = Config(concavity=0.033)
    assert apply_profile(base, get_profile("interactive")) == base


# --- get_profile --------------------------------------------------------


def test_get_profile_default_and_unknown():
    assert get_profile(None).name == "interactive"
    assert get_profile("").name == "interactive"
    with pytest.raises(ValueError, match="unknown profile"):
        get_profile("nonexistent")


def test_all_profiles_resolve():
    for name in PROFILES:
        assert get_profile(name).name == name
