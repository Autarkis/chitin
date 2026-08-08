import pytest

from chitin._metric_names import (
    HULL_COUNT,
    SOURCE_SURFACE_COVERAGE,
    WORST_COMPONENT_SURFACE_COVERAGE,
)
from chitin.acceptance import (
    PROFILES,
    AcceptancePolicy,
    apply_profile,
    evaluate,
    get_profile,
)
from chitin.config import Config

# A clean, fully-covered single-unit build (no per-cell split, no fallback).
CLEAN = {
    HULL_COUNT: 6,
    SOURCE_SURFACE_COVERAGE: 0.99,
    WORST_COMPONENT_SURFACE_COVERAGE: None,
    "fallback_hulls": 0,
    "coacd_deterministic": True,
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


def test_walkable_allows_fallback():
    # A bounding-box fallback is acceptable for a walkable surface.
    assert evaluate(get_profile("walkable").policy, _with(fallback_hulls=2)).passed


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


# --- apply_profile: config precedence -----------------------------------


def test_apply_profile_fills_defaults():
    applied = apply_profile(Config(), get_profile("robotics"))
    assert applied.concavity == 0.01  # preset filled the default
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
