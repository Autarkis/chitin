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

from chitin._metric_names import (
    COLLIDER_VOLUME_PRECISION,
    DEEP_FALSE_FILL_FRACTION,
    FALSE_FILL_FRACTION,
    HULL_COUNT,
    SOURCE_SURFACE_COVERAGE,
    WORST_COMPONENT_SURFACE_COVERAGE,
    WORST_DECILE_SURFACE_COVERAGE,
)
from chitin._shared_constants import PROFILE_NAMES
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
    max_false_fill_fraction: float | None = None
    max_deep_false_fill_fraction: float | None = None
    require_walkable_probe: bool = False
    min_probe_coverage: float | None = None
    max_probe_gap_clusters: int | None = None
    max_compile_ms: float | None = None
    max_output_bytes: int | None = None
    max_hull_vertices: int | None = None


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str
    suggestion: str | None = None

    def to_dict(self) -> dict:
        d = {"check": self.name, "passed": self.passed, "detail": self.detail}
        if self.suggestion is not None:
            d["suggestion"] = self.suggestion
        return d


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
            max_false_fill_fraction=0.50,
            require_walkable_probe=True,
            min_probe_coverage=0.70,
            max_probe_gap_clusters=5,
        ),
    ),
    # Robotics colliders: bounded decomposition and snug fit. Concavity 0.01
    # has a measured runtime cliff even on small concave meshes; 0.05 remains
    # detailed while terminating on the representative corpus. A coarse AABB
    # fallback would put a phantom box around the asset, so a CoACD timeout is
    # disqualifying — the build is rejected rather than shipped. So is a build
    # run with --fast: a collider a simulation is validated against has to be
    # reproducible from its manifest.
    "robotics": Profile(
        name="robotics",
        preset={"concavity": 0.05, "snug_fit": True},
        policy=AcceptancePolicy(
            name="robotics",
            mode="strict",
            require_hulls=True,
            allow_fallback_hulls=False,
            require_deterministic=True,
            min_covered_fraction=0.90,
            min_worst_cell_fraction=0.70,
            max_false_fill_fraction=0.30,
            max_deep_false_fill_fraction=0.20,
        ),
    ),
}

if set(PROFILES) != set(PROFILE_NAMES):
    raise RuntimeError("acceptance profiles do not match the shared profile contract")

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
    probe = detected.get("probe") or {}
    return {
        HULL_COUNT: len(result.hulls),
        SOURCE_SURFACE_COVERAGE: coverage.get(SOURCE_SURFACE_COVERAGE),
        WORST_COMPONENT_SURFACE_COVERAGE: coverage.get(
            WORST_COMPONENT_SURFACE_COVERAGE
        ),
        WORST_DECILE_SURFACE_COVERAGE: coverage.get(WORST_DECILE_SURFACE_COVERAGE),
        FALSE_FILL_FRACTION: coverage.get(FALSE_FILL_FRACTION),
        DEEP_FALSE_FILL_FRACTION: coverage.get(DEEP_FALSE_FILL_FRACTION),
        COLLIDER_VOLUME_PRECISION: coverage.get(COLLIDER_VOLUME_PRECISION),
        "fallback_hulls": int(detected.get("fallback_hulls", 0)),
        "coacd_timeouts": int(detected.get("coacd_timeouts", 0)),
        "coacd_deterministic": detected.get("coacd_deterministic"),
        "probe_coverage": probe.get("coverage") if probe else None,
        "probe_gap_clusters": probe.get("gap_clusters") if probe else None,
        "total_hull_vertices": sum(len(h.vertices) for h in result.hulls),
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
        n = int(metrics.get(HULL_COUNT) or 0)
        ok = n >= 1
        checks.append(
            Check(
                "has_hulls",
                ok,
                f"{n} hull(s) generated" if ok else "no hulls generated",
                suggestion=None
                if ok
                else "Check that input geometry contains valid mesh data",
            )
        )

    if not policy.allow_fallback_hulls:
        fallback = int(metrics.get("fallback_hulls") or 0)
        ok = fallback == 0
        checks.append(
            Check(
                "no_fallback_hulls",
                ok,
                "no CoACD-timeout fallback hulls"
                if ok
                else f"{fallback} AABB fallback hull(s) from CoACD timeout",
                suggestion=None
                if ok
                else "Increase coacd_timeout or simplify the mesh before compiling",
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
                suggestion=None if ok else "Remove --fast flag for reproducible builds",
            )
        )

    if policy.min_covered_fraction is not None:
        covered = metrics.get(SOURCE_SURFACE_COVERAGE)
        ok = covered is not None and covered >= policy.min_covered_fraction
        checks.append(
            Check(
                "coverage",
                ok,
                f"{SOURCE_SURFACE_COVERAGE} {_fmt(covered)} "
                f"{'>=' if ok else '<'} required {policy.min_covered_fraction}",
                suggestion=None
                if ok
                else "Try lowering concavity for finer hulls, or check for "
                "floating geometry",
            )
        )

    if policy.min_worst_cell_fraction is not None:
        worst = metrics.get(WORST_COMPONENT_SURFACE_COVERAGE)
        # Absent on a single-unit build (no per-cell split), which leaves
        # nothing to gate, so it passes.
        ok = worst is None or worst >= policy.min_worst_cell_fraction
        detail = (
            "no per-cell split to gate"
            if worst is None
            else f"{WORST_COMPONENT_SURFACE_COVERAGE} {_fmt(worst)} "
            f"{'>=' if ok else '<'} required {policy.min_worst_cell_fraction}"
        )
        suggestion = (
            None
            if ok or worst is None
            else "Lower concavity or check the component with the worst "
            "coverage for disconnected geometry"
        )
        checks.append(Check("worst_cell_coverage", ok, detail, suggestion=suggestion))

    if policy.max_false_fill_fraction is not None:
        ff = metrics.get(FALSE_FILL_FRACTION)
        ok = ff is None or ff <= policy.max_false_fill_fraction
        checks.append(
            Check(
                "false_fill",
                ok,
                f"false_fill_fraction {_fmt(ff)} {'<=' if ok else '>'} max {policy.max_false_fill_fraction}"
                if ff is not None
                else "volume metrics not measured",
                suggestion=None
                if ok
                else "Lower concavity threshold or enable snug_fit to tighten hull boundaries",
            )
        )

    if policy.max_deep_false_fill_fraction is not None:
        dff = metrics.get(DEEP_FALSE_FILL_FRACTION)
        ok = dff is None or dff <= policy.max_deep_false_fill_fraction
        checks.append(
            Check(
                "deep_false_fill",
                ok,
                f"deep_false_fill_fraction {_fmt(dff)} {'<=' if ok else '>'} max {policy.max_deep_false_fill_fraction}"
                if dff is not None
                else "deep volume metrics not measured",
                suggestion=None
                if ok
                else "Lower concavity threshold; deep false fill means hull boundaries are far from source geometry",
            )
        )

    if policy.require_walkable_probe:
        probe_cov = metrics.get("probe_coverage")
        probe_gaps = metrics.get("probe_gap_clusters")

        if probe_cov is not None and policy.min_probe_coverage is not None:
            ok = probe_cov >= policy.min_probe_coverage
            checks.append(
                Check(
                    "probe_coverage",
                    ok,
                    f"probe coverage {_fmt(probe_cov)} {'>=' if ok else '<'} required {policy.min_probe_coverage}",
                    suggestion=None
                    if ok
                    else "Check for gaps in floor geometry or lower concavity for better coverage",
                )
            )

        if probe_gaps is not None and policy.max_probe_gap_clusters is not None:
            ok = probe_gaps <= policy.max_probe_gap_clusters
            checks.append(
                Check(
                    "probe_gap_clusters",
                    ok,
                    f"{probe_gaps} gap cluster(s) {'<=' if ok else '>'} max {policy.max_probe_gap_clusters}",
                    suggestion=None
                    if ok
                    else "Fill gaps in floor collider geometry to ensure continuous walkable surface",
                )
            )

    if policy.max_compile_ms is not None:
        ms = metrics.get("compile_ms")
        if ms is not None:
            ok = ms <= policy.max_compile_ms
            checks.append(
                Check(
                    "compile_latency",
                    ok,
                    f"compile took {ms:.0f} ms {'<=' if ok else '>'} "
                    f"max {policy.max_compile_ms:.0f} ms",
                    suggestion=None
                    if ok
                    else "Reduce mesh complexity or raise max_compile_ms threshold",
                )
            )

    if policy.max_output_bytes is not None:
        size = metrics.get("output_bytes")
        if size is not None:
            ok = size <= policy.max_output_bytes
            checks.append(
                Check(
                    "output_size",
                    ok,
                    f"output {size:,} bytes {'<=' if ok else '>'} "
                    f"max {policy.max_output_bytes:,} bytes",
                    suggestion=None
                    if ok
                    else "Raise concavity or reduce max_hulls to shrink output",
                )
            )

    if policy.max_hull_vertices is not None:
        verts = metrics.get("total_hull_vertices")
        if verts is not None:
            ok = verts <= policy.max_hull_vertices
            checks.append(
                Check(
                    "hull_vertex_count",
                    ok,
                    f"{verts:,} hull vertices {'<=' if ok else '>'} "
                    f"max {policy.max_hull_vertices:,}",
                    suggestion=None
                    if ok
                    else "Lower max_hull_vertices in config or raise concavity",
                )
            )

    passed = all(c.passed for c in checks)
    return Verdict(profile=policy.name, passed=passed, checks=tuple(checks))
