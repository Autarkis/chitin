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
    WORST_DECILE_SURFACE_COVERAGE,
)
from chitin._shared_constants import ACCEPTANCE_THRESHOLDS, PROFILE_NAMES
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
    # DEPRECATED: new builds never produce fallback hulls (chitin #102).
    # Retained for interpreting historical artifacts.
    allow_fallback_hulls: bool = True
    require_deterministic: bool = False
    require_snug_fit: bool = False
    min_covered_fraction: float | None = None
    min_worst_cell_fraction: float | None = None
    max_false_fill_fraction: float | None = None
    max_deep_false_fill_fraction: float | None = None
    # DEPRECATED: see allow_fallback_hulls.
    max_fallback_ratio: float | None = None
    require_walkable_probe: bool = False
    min_probe_coverage: float | None = None
    max_probe_gap_clusters: int | None = None
    require_walkable_sweep: bool = False
    min_sweep_traversability: float | None = None
    min_standable_fraction: float | None = None
    max_clearance_blocked_fraction: float | None = None
    max_compile_ms: float | None = None
    max_output_bytes: int | None = None
    max_hull_count: int | None = None
    max_hull_vertices: int | None = None
    max_hull_triangles: int | None = None


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


_WALKABLE_THRESHOLDS = ACCEPTANCE_THRESHOLDS["walkable"]
_ROBOTICS_THRESHOLDS = ACCEPTANCE_THRESHOLDS["robotics"]

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
            max_fallback_ratio=_WALKABLE_THRESHOLDS["max_fallback_ratio"],
            min_covered_fraction=_WALKABLE_THRESHOLDS["min_covered_fraction"],
            max_false_fill_fraction=_WALKABLE_THRESHOLDS["max_false_fill_fraction"],
            require_walkable_probe=True,
            min_probe_coverage=_WALKABLE_THRESHOLDS["min_probe_coverage"],
            max_probe_gap_clusters=_WALKABLE_THRESHOLDS["max_probe_gap_clusters"],
            require_walkable_sweep=True,
            min_sweep_traversability=_WALKABLE_THRESHOLDS["min_sweep_traversability"],
            min_standable_fraction=_WALKABLE_THRESHOLDS["min_standable_fraction"],
            max_clearance_blocked_fraction=_WALKABLE_THRESHOLDS[
                "max_clearance_blocked_fraction"
            ],
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
            require_snug_fit=True,
            min_covered_fraction=_ROBOTICS_THRESHOLDS["min_covered_fraction"],
            min_worst_cell_fraction=_ROBOTICS_THRESHOLDS["min_worst_cell_fraction"],
            max_false_fill_fraction=_ROBOTICS_THRESHOLDS["max_false_fill_fraction"],
            max_deep_false_fill_fraction=_ROBOTICS_THRESHOLDS[
                "max_deep_false_fill_fraction"
            ],
            max_hull_count=_ROBOTICS_THRESHOLDS["max_hull_count"],
            max_hull_vertices=_ROBOTICS_THRESHOLDS["max_hull_vertices"],
            max_hull_triangles=_ROBOTICS_THRESHOLDS["max_hull_triangles"],
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


def record_artifact_checks(result, policy: AcceptancePolicy) -> None:
    """Run the in-memory artifact checks required by ``policy``.

    Results are stored on the build plan so the verdict, canonical report, and
    provenance manifest all consume the same measurements. Required checks are
    deliberately left absent when there is no plan or no hull artifact; strict
    evaluation then reports that missing processing as a failed check.
    """
    plan = result.build_plan
    if plan is None or not result.hulls:
        return

    if policy.require_walkable_probe:
        from chitin.verify.probe import probe_from_hulls

        probe = probe_from_hulls(result.hulls, grid_resolution=32)
        plan.detected["probe"] = {
            "coverage": probe.coverage,
            "gap_clusters": probe.gap_clusters,
            "hits": probe.hits,
            "misses": probe.misses,
            "total_rays": probe.total_rays,
            "confidence": probe.confidence,
        }

    if policy.require_walkable_sweep:
        from chitin.verify.sweep import sweep_hulls

        sweep = sweep_hulls(result.hulls, grid_resolution=32)
        denominator = sweep.ground_cells
        plan.detected["sweep"] = {
            "traversability": sweep.traversability,
            "standable_fraction": (
                sweep.standable_cells / denominator if denominator else 0.0
            ),
            "clearance_blocked_fraction": (
                sweep.clearance_blocked / denominator if denominator else 0.0
            ),
            "radius_blocked_fraction": (
                sweep.radius_blocked / denominator if denominator else 0.0
            ),
            "ground_cells": sweep.ground_cells,
            "standable_cells": sweep.standable_cells,
            "clearance_blocked": sweep.clearance_blocked,
            "radius_blocked": sweep.radius_blocked,
            "connected_components": sweep.connected_components,
            "largest_component": sweep.largest_component,
            "seam_snags": sweep.seam_snags,
            "capsule_radius": sweep.capsule_radius,
            "capsule_height": sweep.capsule_height,
            "step_height": sweep.step_height,
        }


def report_metrics(result) -> dict:
    """Flatten the quality signals :func:`evaluate` reads out of a result."""
    plan = result.build_plan
    detected = plan.detected if plan is not None else {}
    coverage = detected.get("coverage") or {}
    probe = detected.get("probe") or {}
    sweep = detected.get("sweep") or {}
    hull_count = len(result.hulls)
    fallback_hulls = int(detected.get("fallback_hulls", 0))
    snug_fit_requested = getattr(result.resolved, "snug_fit", None)
    snug_fit_stats_present = any(
        key in detected
        for key in ("snugfit_refined", "snugfit_rejected", "snugfit_skipped")
    )
    if snug_fit_requested is False:
        snug_fit_status = "not_requested"
    elif snug_fit_stats_present:
        snug_fit_status = "applied"
    elif snug_fit_requested is True:
        snug_fit_status = "skipped"
    else:
        snug_fit_status = "unknown"
    return {
        HULL_COUNT: hull_count,
        SOURCE_SURFACE_COVERAGE: coverage.get(SOURCE_SURFACE_COVERAGE),
        WORST_COMPONENT_SURFACE_COVERAGE: coverage.get(
            WORST_COMPONENT_SURFACE_COVERAGE
        ),
        WORST_DECILE_SURFACE_COVERAGE: coverage.get(WORST_DECILE_SURFACE_COVERAGE),
        FALSE_FILL_FRACTION: coverage.get(FALSE_FILL_FRACTION),
        DEEP_FALSE_FILL_FRACTION: coverage.get(DEEP_FALSE_FILL_FRACTION),
        COLLIDER_VOLUME_PRECISION: coverage.get(COLLIDER_VOLUME_PRECISION),
        "fallback_hulls": fallback_hulls,
        FALLBACK_RATIO: fallback_hulls / hull_count if hull_count else 0.0,
        PLANAR_SUBSTITUTE_HULLS: int(detected.get("planar_substitute_hulls", 0)),
        "coacd_timeouts": int(detected.get("coacd_timeouts", 0)),
        "coacd_deterministic": detected.get("coacd_deterministic"),
        SNUG_FIT_STATUS: snug_fit_status,
        PROBE_COVERAGE: probe.get("coverage") if probe else None,
        PROBE_GAP_CLUSTERS: probe.get("gap_clusters") if probe else None,
        SWEEP_TRAVERSABILITY: sweep.get("traversability") if sweep else None,
        STANDABLE_FRACTION: sweep.get("standable_fraction") if sweep else None,
        CLEARANCE_BLOCKED_FRACTION: (
            sweep.get("clearance_blocked_fraction") if sweep else None
        ),
        RADIUS_BLOCKED_FRACTION: (
            sweep.get("radius_blocked_fraction") if sweep else None
        ),
        SEAM_SNAG_COUNT: len(sweep.get("seam_snags", ())) if sweep else None,
        HULL_VERTEX_COUNT: sum(len(h.vertices) for h in result.hulls),
        HULL_TRIANGLE_COUNT: sum(len(h.indices) // 3 for h in result.hulls),
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

    # Both checks below are effectively dead code for new builds: a CoACD
    # timeout now raises CompilationError before a build_plan even exists
    # (chitin #102), so fallback_hulls is always 0. Retained so a historical
    # artifact's build_plan (from before #102) still evaluates correctly.
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

    if policy.max_fallback_ratio is not None:
        ratio = metrics.get(FALLBACK_RATIO)
        ok = ratio is not None and ratio <= policy.max_fallback_ratio
        checks.append(
            Check(
                "fallback_ratio",
                ok,
                f"fallback_ratio {_fmt(ratio)} "
                f"{'<=' if ok else '>'} max {policy.max_fallback_ratio}"
                if ratio is not None
                else "fallback ratio not measured",
                suggestion=None
                if ok
                else "Increase the CoACD timeout or simplify the mesh to avoid "
                "decomposition-failure bounding boxes",
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

    if policy.require_snug_fit:
        status = metrics.get(SNUG_FIT_STATUS)
        ok = status == "applied"
        checks.append(
            Check(
                "snug_fit_applied",
                ok,
                "snug-fit refinement applied"
                if ok
                else f"snug-fit refinement {status or 'not measured'}",
                suggestion=None
                if ok
                else "Install the snug-fit dependencies and rerun without skipping "
                "the requested refinement",
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
        ok = ff is not None and ff <= policy.max_false_fill_fraction
        checks.append(
            Check(
                "false_fill",
                ok,
                f"false_fill_fraction {_fmt(ff)} {'<=' if ok else '>'} max {policy.max_false_fill_fraction}"
                if ff is not None
                else "volume metrics not measured",
                suggestion=(
                    None
                    if ok
                    else "Run volume verification before evaluating this profile"
                    if ff is None
                    else "Lower concavity threshold or enable snug_fit to tighten "
                    "hull boundaries"
                ),
            )
        )

    if policy.max_deep_false_fill_fraction is not None:
        dff = metrics.get(DEEP_FALSE_FILL_FRACTION)
        ok = dff is not None and dff <= policy.max_deep_false_fill_fraction
        checks.append(
            Check(
                "deep_false_fill",
                ok,
                f"deep_false_fill_fraction {_fmt(dff)} {'<=' if ok else '>'} max {policy.max_deep_false_fill_fraction}"
                if dff is not None
                else "deep volume metrics not measured",
                suggestion=(
                    None
                    if ok
                    else "Run volume verification before evaluating this profile"
                    if dff is None
                    else "Lower concavity threshold; deep false fill means hull "
                    "boundaries are far from source geometry"
                ),
            )
        )

    if policy.require_walkable_probe:
        probe_cov = metrics.get(PROBE_COVERAGE)
        probe_gaps = metrics.get(PROBE_GAP_CLUSTERS)

        if policy.min_probe_coverage is not None:
            ok = probe_cov is not None and probe_cov >= policy.min_probe_coverage
            checks.append(
                Check(
                    "probe_coverage",
                    ok,
                    f"probe coverage {_fmt(probe_cov)} {'>=' if ok else '<'} required {policy.min_probe_coverage}",
                    suggestion=(
                        None
                        if ok
                        else "Run the walkable artifact probe before evaluating this profile"
                        if probe_cov is None
                        else "Check for gaps in floor geometry or lower concavity for "
                        "better coverage"
                    ),
                )
            )

        if policy.max_probe_gap_clusters is not None:
            ok = probe_gaps is not None and probe_gaps <= policy.max_probe_gap_clusters
            checks.append(
                Check(
                    "probe_gap_clusters",
                    ok,
                    f"{probe_gaps} gap cluster(s) "
                    f"{'<=' if ok else '>'} max {policy.max_probe_gap_clusters}"
                    if probe_gaps is not None
                    else "probe gap clusters not measured",
                    suggestion=(
                        None
                        if ok
                        else "Run the walkable artifact probe before evaluating this profile"
                        if probe_gaps is None
                        else "Fill gaps in floor collider geometry to ensure a "
                        "continuous walkable surface"
                    ),
                )
            )

    if policy.require_walkable_sweep:
        traversability = metrics.get(SWEEP_TRAVERSABILITY)
        standable = metrics.get(STANDABLE_FRACTION)
        clearance = metrics.get(CLEARANCE_BLOCKED_FRACTION)

        if policy.min_sweep_traversability is not None:
            ok = (
                traversability is not None
                and traversability >= policy.min_sweep_traversability
            )
            checks.append(
                Check(
                    "capsule_traversability",
                    ok,
                    f"capsule traversability {_fmt(traversability)} "
                    f"{'>=' if ok else '<'} required "
                    f"{policy.min_sweep_traversability}",
                    suggestion=(
                        None
                        if ok
                        else "Run the capsule sweep before evaluating this profile"
                        if traversability is None
                        else "Repair disconnected floor islands or step-height seam snags"
                    ),
                )
            )

        if policy.min_standable_fraction is not None:
            ok = standable is not None and standable >= policy.min_standable_fraction
            checks.append(
                Check(
                    "capsule_standable_fraction",
                    ok,
                    f"standable fraction {_fmt(standable)} "
                    f"{'>=' if ok else '<'} required {policy.min_standable_fraction}",
                    suggestion=(
                        None
                        if ok
                        else "Run the capsule sweep before evaluating this profile"
                        if standable is None
                        else "Widen narrow passages or remove lateral collider overfill"
                    ),
                )
            )

        if policy.max_clearance_blocked_fraction is not None:
            ok = (
                clearance is not None
                and clearance <= policy.max_clearance_blocked_fraction
            )
            checks.append(
                Check(
                    "capsule_clearance",
                    ok,
                    f"clearance-blocked fraction {_fmt(clearance)} "
                    f"{'<=' if ok else '>'} max "
                    f"{policy.max_clearance_blocked_fraction}",
                    suggestion=(
                        None
                        if ok
                        else "Run the capsule sweep before evaluating this profile"
                        if clearance is None
                        else "Increase headroom or remove collider geometry above the floor"
                    ),
                )
            )

    if policy.max_compile_ms is not None:
        ms = metrics.get("compile_ms")
        ok = ms is not None and ms <= policy.max_compile_ms
        checks.append(
            Check(
                "compile_latency",
                ok,
                f"compile took {ms:.0f} ms {'<=' if ok else '>'} "
                f"max {policy.max_compile_ms:.0f} ms"
                if ms is not None
                else "compile latency not measured",
                suggestion=None
                if ok
                else "Measure compile latency, reduce mesh complexity, or raise "
                "the configured threshold",
            )
        )

    if policy.max_output_bytes is not None:
        size = metrics.get("output_bytes")
        ok = size is not None and size <= policy.max_output_bytes
        checks.append(
            Check(
                "output_size",
                ok,
                f"output {size:,} bytes {'<=' if ok else '>'} "
                f"max {policy.max_output_bytes:,} bytes"
                if size is not None
                else "output size not measured",
                suggestion=None
                if ok
                else "Measure the artifact size, raise concavity, or reduce max_hulls",
            )
        )

    if policy.max_hull_count is not None:
        hulls = metrics.get(HULL_COUNT)
        ok = hulls is not None and hulls <= policy.max_hull_count
        checks.append(
            Check(
                "hull_count",
                ok,
                f"{hulls:,} hulls {'<=' if ok else '>'} max {policy.max_hull_count:,}"
                if hulls is not None
                else "hull count not measured",
                suggestion=None
                if ok
                else "Raise concavity or reduce max_hulls to lower collider complexity",
            )
        )

    if policy.max_hull_vertices is not None:
        verts = metrics.get(HULL_VERTEX_COUNT)
        ok = verts is not None and verts <= policy.max_hull_vertices
        checks.append(
            Check(
                "hull_vertex_count",
                ok,
                f"{verts:,} hull vertices {'<=' if ok else '>'} "
                f"max {policy.max_hull_vertices:,}"
                if verts is not None
                else "hull vertex count not measured",
                suggestion=None
                if ok
                else "Lower per-hull vertex detail or raise concavity",
            )
        )

    if policy.max_hull_triangles is not None:
        triangles = metrics.get(HULL_TRIANGLE_COUNT)
        ok = triangles is not None and triangles <= policy.max_hull_triangles
        checks.append(
            Check(
                "hull_triangle_count",
                ok,
                f"{triangles:,} hull triangles {'<=' if ok else '>'} "
                f"max {policy.max_hull_triangles:,}"
                if triangles is not None
                else "hull triangle count not measured",
                suggestion=None
                if ok
                else "Lower per-hull vertex detail or raise concavity",
            )
        )

    passed = all(c.passed for c in checks)
    return Verdict(profile=policy.name, passed=passed, checks=tuple(checks))
