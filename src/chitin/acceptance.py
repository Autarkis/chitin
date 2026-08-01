"""Build profiles and artifact acceptance.

``interactive``, ``walkable``, and ``robotics`` used to be a label carried on a
job and never read, so all three produced byte-identical builds. Here each is a
real :class:`Profile` — a small set of config presets plus an
:class:`AcceptancePolicy` — and :func:`evaluate` turns a build's measured
metrics into a pass/fail :class:`Verdict` with reasons. The verdict is surfaced
in ``report.json`` and the provenance ``manifest.json``; a failed strict build
is *rejected*, not silently completed.

:func:`evaluate` is a pure function of ``(policy, metrics)`` so it can be
unit-tested against synthetic reports with no pipeline run.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, replace

from chitin.config import Config


@dataclass(frozen=True)
class AcceptancePolicy:
    """Thresholds a build must clear to be accepted under a profile.

    A ``None`` threshold is not checked. ``permissive`` policies leave every
    threshold unset and therefore always pass; they exist so an interactive
    build never newly fails.
    """

    name: str
    mode: str = "permissive"  # "permissive" | "strict"
    require_hulls: bool = False
    allow_fallback_hulls: bool = True
    require_deterministic: bool = False
    min_covered_fraction: float | None = None
    min_worst_cell_fraction: float | None = None


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return {"check": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class Verdict:
    profile: str
    passed: bool
    checks: tuple[Check, ...] = ()

    @property
    def reasons(self) -> list[str]:
        """Human-readable details of the checks that failed."""
        return [c.detail for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {
            "profile": self.profile,
            "passed": self.passed,
            "reasons": self.reasons,
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass(frozen=True)
class Profile:
    """A named build preset: config defaults plus an acceptance policy.

    ``preset`` maps :class:`~chitin.config.Config` field names to values that
    fill in only where the caller did not set the field (see
    :func:`apply_profile`), so an explicit ``--concavity`` always wins over the
    profile.
    """

    name: str
    preset: dict
    policy: AcceptancePolicy


PROFILES: dict[str, Profile] = {
    # Permissive default: geometry unchanged, no acceptance gates. A build that
    # completes today keeps completing.
    "interactive": Profile(
        name="interactive",
        preset={},
        policy=AcceptancePolicy(name="interactive", mode="permissive"),
    ),
    # Environment scans: coarser decomposition, denser Poisson filtering. A
    # bounding-box fallback is acceptable for a walkable floor plate, but the
    # scene must be substantially covered.
    "walkable": Profile(
        name="walkable",
        preset={"concavity": 0.1, "poisson_density_quantile": 0.3},
        policy=AcceptancePolicy(
            name="walkable",
            mode="strict",
            require_hulls=True,
            allow_fallback_hulls=True,
            min_covered_fraction=0.85,
        ),
    ),
    # Robotics colliders: tight decomposition and snug fit. A coarse AABB
    # fallback would put a phantom box around the asset, so a CoACD timeout is
    # disqualifying — the build is rejected rather than shipped. So is a build
    # run with --fast: a collider a simulation is validated against has to be
    # reproducible from its manifest.
    "robotics": Profile(
        name="robotics",
        preset={"concavity": 0.01, "snug_fit": True},
        policy=AcceptancePolicy(
            name="robotics",
            mode="strict",
            require_hulls=True,
            allow_fallback_hulls=False,
            require_deterministic=True,
            min_covered_fraction=0.90,
            min_worst_cell_fraction=0.70,
        ),
    ),
}

DEFAULT_PROFILE = "interactive"


def get_profile(name: str | None) -> Profile:
    if not name:
        return PROFILES[DEFAULT_PROFILE]
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown profile {name!r}; choose from {sorted(PROFILES)}"
        ) from None


def apply_profile(
    config: Config, profile: Profile, explicit: Collection[str] | None = None
) -> Config:
    """Layer a profile's presets onto ``config`` without clobbering intent.

    A preset fills a field only where the caller did not set it, so an
    explicitly customized value always wins over the profile's suggestion.

    ``explicit`` names the fields the caller actually supplied. Without it the
    function has to infer that by comparing against ``Config()``, which cannot
    tell ``--concavity 0.05`` (the default value, deliberately chosen) from no
    flag at all and would overwrite it; pass the set whenever the caller knows.
    """
    if not profile.preset:
        return config
    if explicit is None:
        default = Config()
        explicit = {
            field
            for field in profile.preset
            if getattr(config, field) != getattr(default, field)
        }
    overrides = {
        field: value for field, value in profile.preset.items() if field not in explicit
    }
    return replace(config, **overrides) if overrides else config


def report_metrics(result) -> dict:
    """Flatten the quality signals :func:`evaluate` reads out of a result."""
    plan = result.build_plan
    detected = plan.detected if plan is not None else {}
    coverage = detected.get("coverage") or {}
    return {
        "hull_count": len(result.hulls),
        "covered_fraction": coverage.get("covered_fraction"),
        "worst_cell_fraction": coverage.get("worst_cell_fraction"),
        "worst_decile_fraction": coverage.get("worst_decile_fraction"),
        "fallback_hulls": int(detected.get("fallback_hulls", 0)),
        "coacd_timeouts": int(detected.get("coacd_timeouts", 0)),
        "coacd_deterministic": detected.get("coacd_deterministic"),
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def evaluate(policy: AcceptancePolicy, metrics: dict) -> Verdict:
    """Turn measured ``metrics`` into a pass/fail verdict under ``policy``.

    Pure function: ``metrics`` is a flat dict (see :func:`report_metrics`), so
    the same call can be exercised against synthetic reports in tests.
    """
    checks: list[Check] = []

    if policy.require_hulls:
        n = int(metrics.get("hull_count") or 0)
        checks.append(
            Check(
                "has_hulls",
                n >= 1,
                f"{n} hull(s) generated" if n >= 1 else "no hulls generated",
            )
        )

    if not policy.allow_fallback_hulls:
        fallback = int(metrics.get("fallback_hulls") or 0)
        checks.append(
            Check(
                "no_fallback_hulls",
                fallback == 0,
                "no CoACD-timeout fallback hulls"
                if fallback == 0
                else f"{fallback} AABB fallback hull(s) from CoACD timeout",
            )
        )

    if policy.require_deterministic:
        flag = metrics.get("coacd_deterministic")
        # Absent means no CoACD ran at all (a planar box, an all-environment
        # build), so there is no unreproducible search to reject.
        ok = flag is None or bool(flag)
        checks.append(
            Check(
                "deterministic_decomposition",
                ok,
                "CoACD ran single-threaded (reproducible)"
                if ok
                else "CoACD ran multithreaded (--fast): hulls vary run to run "
                "and cannot be reproduced from this manifest",
            )
        )

    if policy.min_covered_fraction is not None:
        covered = metrics.get("covered_fraction")
        ok = covered is not None and covered >= policy.min_covered_fraction
        checks.append(
            Check(
                "coverage",
                ok,
                f"covered_fraction {_fmt(covered)} "
                f"{'>=' if ok else '<'} required {policy.min_covered_fraction}",
            )
        )

    if policy.min_worst_cell_fraction is not None:
        worst = metrics.get("worst_cell_fraction")
        # Absent on a single-unit build (no per-cell split), which leaves
        # nothing to gate, so it passes.
        ok = worst is None or worst >= policy.min_worst_cell_fraction
        detail = (
            "no per-cell split to gate"
            if worst is None
            else f"worst_cell_fraction {_fmt(worst)} "
            f"{'>=' if ok else '<'} required {policy.min_worst_cell_fraction}"
        )
        checks.append(Check("worst_cell_coverage", ok, detail))

    passed = all(c.passed for c in checks)
    return Verdict(profile=policy.name, passed=passed, checks=tuple(checks))
