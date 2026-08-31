"""Versioned, cross-runtime compilation report contract.

``CompilationReport`` is the canonical JSON shape shared with
``@autarkis/chitin-lite``. The service embeds this object under
``compilation_report`` alongside the flat ``report.json`` fields.
"""

from __future__ import annotations

import functools
import importlib.resources
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chitin import provenance
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

_PACKAGED_SCHEMA = importlib.resources.files("chitin").joinpath(
    "compilation-report.schema.json"
)
_SOURCE_SCHEMA = (
    Path(__file__).resolve().parent.parent.parent
    / "docs"
    / "compilation-report.schema.json"
)

REPORT_VERSION = 1
REPORT_FIELDS = (
    "report_version",
    "status",
    "profile",
    "verdict",
    "input",
    "output",
    "timings_ms",
    "warnings",
    "metrics",
    "processing",
    "runtime",
    "reproducibility",
    "config",
    "artifacts",
)


@dataclass(frozen=True)
class ReportWarning:
    code: str
    message: str
    severity: str = "warning"
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "context": dict(self.context),
        }


@dataclass(frozen=True)
class ReportMetric:
    value: bool | int | float | str | None
    unit: str
    status: str = "measured"

    def to_dict(self) -> dict:
        return {"value": self.value, "unit": self.unit, "status": self.status}


@dataclass(frozen=True)
class CompilationReport:
    status: str
    profile: str | None
    verdict: dict
    input: dict
    output: dict
    timings_ms: dict[str, float]
    warnings: tuple[ReportWarning, ...]
    metrics: dict[str, ReportMetric]
    processing: dict
    runtime: dict
    reproducibility: dict
    config: dict
    artifacts: dict[str, str]
    report_version: int = REPORT_VERSION

    def to_dict(self) -> dict:
        return {
            "report_version": self.report_version,
            "status": self.status,
            "profile": self.profile,
            "verdict": dict(self.verdict),
            "input": dict(self.input),
            "output": dict(self.output),
            "timings_ms": dict(self.timings_ms),
            "warnings": [warning.to_dict() for warning in self.warnings],
            "metrics": {
                name: metric.to_dict() for name, metric in self.metrics.items()
            },
            "processing": dict(self.processing),
            "runtime": dict(self.runtime),
            "reproducibility": dict(self.reproducibility),
            "config": dict(self.config),
            "artifacts": dict(self.artifacts),
        }


def _metric(
    value: bool | float | str | None,
    unit: str,
    *,
    absent: str = "not_measured",
) -> ReportMetric:
    return ReportMetric(
        value=value,
        unit=unit,
        status="measured" if value is not None else absent,
    )


def _verdict_dict(profile: str | None, verdict) -> dict:
    if verdict is None:
        return {
            "profile": profile,
            "status": "not_evaluated",
            "reasons": [],
            "checks": [],
        }
    checks_list = []
    for check in verdict.checks:
        entry = {
            "code": check.name,
            "status": "pass" if check.passed else "fail",
            "message": check.detail,
        }
        if check.suggestion is not None:
            entry["suggestion"] = check.suggestion
        checks_list.append(entry)
    return {
        "profile": verdict.profile,
        "status": "pass" if verdict.passed else "fail",
        "reasons": verdict.reasons,
        "checks": checks_list,
    }


def detected_issues(result) -> list[ReportWarning]:
    """Non-fatal quality issues derived from ``result.build_plan.detected``.

    This is the single source of truth for what counts as a build-plan
    warning; both :class:`CompilationReport` (via :func:`_report_warnings`)
    and :func:`chitin.manifest.quality_warnings` derive from it so a new
    ``detected`` key only needs to be taught here once.
    """
    plan = result.build_plan
    detected = plan.detected if plan is not None else {}
    issues: list[ReportWarning] = []
    fallback = int(detected.get("fallback_hulls", 0))
    if fallback:
        issues.append(
            ReportWarning(
                code="COACD_TIMEOUT_FALLBACK",
                message=f"{fallback} AABB fallback hull(s) from CoACD timeout",
                context={"hull_count": fallback},
            )
        )
    if plan is not None and plan.decimated:
        issues.append(
            ReportWarning(
                code="INPUT_DECIMATED",
                message="mesh was decimated before decomposition",
            )
        )
    if detected.get("decimation_skipped"):
        issues.append(
            ReportWarning(
                code="DECIMATION_SKIPPED",
                message=(
                    "mesh exceeded max_decompose_vertices but decimation was "
                    "skipped (Open3D not installed)"
                ),
                context={"vertex_count": int(detected["decimation_skipped"])},
            )
        )
    if detected.get("bones_skipped"):
        issues.append(
            ReportWarning(
                code="BONES_SKIPPED",
                message=(
                    f"{detected['bones_skipped']} bones had too little geometry "
                    "for hull generation"
                ),
                context={"bone_count": int(detected["bones_skipped"])},
            )
        )
    return issues


def _report_warnings(result, extra: list[str] | None) -> tuple[ReportWarning, ...]:
    warnings = detected_issues(result)
    known_messages = {warning.message for warning in warnings}
    for message in extra or []:
        if message not in known_messages:
            warnings.append(ReportWarning(code="PIPELINE_WARNING", message=message))
    return tuple(warnings)


def _snug_fit_status(detected: dict, effective_config: dict | None) -> dict:
    requested = (
        None if effective_config is None else bool(effective_config.get("snug_fit"))
    )
    stats_present = any(
        key in detected
        for key in ("snugfit_refined", "snugfit_rejected", "snugfit_skipped")
    )
    if requested is False:
        status = "not_requested"
    elif stats_present:
        status = "applied"
    elif requested is True:
        # This includes the current no-scipy path. Calling it out is deliberate:
        # a profile must not silently promise a refinement that did not run.
        status = "skipped"
    else:
        status = "unknown"
    return {
        "status": status,
        "refined_hulls": detected.get("snugfit_refined"),
        "rejected_hulls": detected.get("snugfit_rejected"),
        "skipped_hulls": detected.get("snugfit_skipped"),
    }


def select_primary_artifact(paths: Iterable[Path]) -> Path | None:
    """Pick the one artifact whose size and hash represent this build.

    Preference order is fixed -- ``.phys``, then ``.usda``/``.usd``, then the
    lexicographically-first remaining file -- so the same set of written
    files always selects the same primary regardless of write order or which
    output formats a caller happened to request. Returns ``None`` if
    ``paths`` contains no files.
    """
    files = [p for p in paths if p.is_file()]
    if not files:
        return None

    def _rank(path: Path) -> tuple[int, str]:
        suffix = path.suffix.lstrip(".")
        if suffix == "phys":
            tier = 0
        elif suffix in ("usda", "usd"):
            tier = 1
        else:
            tier = 2
        return (tier, path.name)

    return min(files, key=_rank)


def build_compilation_report(
    result,
    *,
    profile: str | None = None,
    verdict=None,
    warnings: list[str] | None = None,
    requested_config: dict | None = None,
    effective_config: dict | None = None,
    artifacts: dict[str, str] | None = None,
    artifact_bytes: int | None = None,
    artifact_sha256: str | None = None,
    timings_ms: dict[str, float] | None = None,
) -> CompilationReport:
    """Build the canonical report without changing acceptance behavior."""
    from chitin.acceptance import report_metrics

    plan = result.build_plan
    detected = plan.detected if plan is not None else {}
    flat_metrics = report_metrics(result)
    resolved_profile = verdict.profile if verdict is not None else profile
    verdict_dict = _verdict_dict(resolved_profile, verdict)

    hull_vertices = sum(len(hull.vertices) for hull in result.hulls)
    hull_triangles = sum(len(hull.indices) // 3 for hull in result.hulls)
    metrics = {
        HULL_COUNT: _metric(flat_metrics[HULL_COUNT], "count"),
        SOURCE_SURFACE_COVERAGE: _metric(
            flat_metrics[SOURCE_SURFACE_COVERAGE], "ratio"
        ),
        WORST_COMPONENT_SURFACE_COVERAGE: _metric(
            flat_metrics[WORST_COMPONENT_SURFACE_COVERAGE], "ratio"
        ),
        WORST_DECILE_SURFACE_COVERAGE: _metric(
            flat_metrics[WORST_DECILE_SURFACE_COVERAGE], "ratio"
        ),
        FALSE_FILL_FRACTION: _metric(flat_metrics.get(FALSE_FILL_FRACTION), "ratio"),
        DEEP_FALSE_FILL_FRACTION: _metric(
            flat_metrics.get(DEEP_FALSE_FILL_FRACTION), "ratio"
        ),
        COLLIDER_VOLUME_PRECISION: _metric(
            flat_metrics.get(COLLIDER_VOLUME_PRECISION), "ratio"
        ),
        FALLBACK_RATIO: _metric(flat_metrics.get(FALLBACK_RATIO), "ratio"),
        PLANAR_SUBSTITUTE_HULLS: _metric(
            flat_metrics.get(PLANAR_SUBSTITUTE_HULLS), "count"
        ),
        HULL_VERTEX_COUNT: _metric(flat_metrics.get(HULL_VERTEX_COUNT), "count"),
        HULL_TRIANGLE_COUNT: _metric(flat_metrics.get(HULL_TRIANGLE_COUNT), "count"),
        PROBE_COVERAGE: _metric(flat_metrics.get(PROBE_COVERAGE), "ratio"),
        PROBE_GAP_CLUSTERS: _metric(flat_metrics.get(PROBE_GAP_CLUSTERS), "count"),
        SWEEP_TRAVERSABILITY: _metric(flat_metrics.get(SWEEP_TRAVERSABILITY), "ratio"),
        STANDABLE_FRACTION: _metric(flat_metrics.get(STANDABLE_FRACTION), "ratio"),
        CLEARANCE_BLOCKED_FRACTION: _metric(
            flat_metrics.get(CLEARANCE_BLOCKED_FRACTION), "ratio"
        ),
        RADIUS_BLOCKED_FRACTION: _metric(
            flat_metrics.get(RADIUS_BLOCKED_FRACTION), "ratio"
        ),
        SEAM_SNAG_COUNT: _metric(flat_metrics.get(SEAM_SNAG_COUNT), "count"),
        SNUG_FIT_STATUS: _metric(flat_metrics.get(SNUG_FIT_STATUS), "status"),
        "fallback_hulls": _metric(flat_metrics["fallback_hulls"], "count"),
        "coacd_timeouts": _metric(flat_metrics["coacd_timeouts"], "count"),
        "coacd_deterministic": _metric(
            flat_metrics["coacd_deterministic"],
            "boolean",
            absent="not_applicable",
        ),
    }

    report = CompilationReport(
        status="rejected" if verdict is not None and not verdict.passed else "complete",
        profile=resolved_profile,
        verdict=verdict_dict,
        input={
            "kind": plan.input_kind if plan is not None else "unknown",
            "source_vertices": int(result.source_vertex_count),
            "processed_vertices": int(
                plan.processed_vertices
                if plan is not None
                else result.mesh_vertex_count
            ),
            "mesh_vertices": int(result.mesh_vertex_count),
        },
        output={
            "collider_kind": plan.collider_kind if plan is not None else "unknown",
            "hull_count": len(result.hulls),
            "vertex_count": hull_vertices,
            "triangle_count": hull_triangles,
            "lod_tier_count": len(result.lod_tiers or []),
            "byte_length": artifact_bytes,
        },
        timings_ms={name: float(value) for name, value in (timings_ms or {}).items()},
        warnings=_report_warnings(result, warnings),
        metrics=metrics,
        processing={
            "pipeline": list(plan.pipeline) if plan is not None else [],
            "fallbacks": {
                "decomposition_failure_hulls": int(detected.get("fallback_hulls", 0)),
                "planar_substitute_hulls": int(
                    detected.get("planar_substitute_hulls", 0)
                ),
            },
            "refinements": {
                "snug_fit": _snug_fit_status(detected, effective_config),
            },
        },
        runtime={
            "kind": "python_native",
            "implementation": "chitin",
            "version": provenance.base_version(),
            "compiler_version": provenance.compiler_version(),
            "dependencies": provenance.dependency_versions(),
        },
        reproducibility={
            "scope": "same_runtime_toolchain",
            "deterministic": flat_metrics["coacd_deterministic"],
            "artifact_sha256": artifact_sha256,
        },
        config={
            "requested": requested_config,
            "effective": effective_config,
        },
        artifacts=dict(artifacts or {}),
    )
    problems = validate_compilation_report(report.to_dict())
    if problems:
        raise ValueError("invalid compilation report: " + "; ".join(problems))
    return report


@functools.lru_cache(maxsize=1)
def _load_schema() -> dict:
    try:
        payload = _PACKAGED_SCHEMA.read_text(encoding="utf-8")
    except FileNotFoundError:
        payload = _SOURCE_SCHEMA.read_text(encoding="utf-8")
    return json.loads(payload)


def validate_compilation_report(report: dict) -> list[str]:
    """Return schema violations, or structural problems without ``jsonschema``."""
    try:
        import jsonschema
    except ImportError:
        return _validate_compilation_report_hand_rolled(report)

    validator = jsonschema.Draft202012Validator(_load_schema())
    problems = []
    for error in validator.iter_errors(report):
        path = ".".join(str(part) for part in error.absolute_path)
        problems.append(f"{path}: {error.message}" if path else error.message)
    return problems


def _validate_compilation_report_hand_rolled(report: dict) -> list[str]:
    """Validate the report shape without optional dependencies."""
    problems: list[str] = []
    missing = [field for field in REPORT_FIELDS if field not in report]
    if missing:
        problems.append(f"missing fields: {', '.join(missing)}")
        return problems
    if report["report_version"] != REPORT_VERSION:
        problems.append(
            f"unsupported report_version {report['report_version']}; "
            f"expected {REPORT_VERSION}"
        )
    if report["status"] not in {"complete", "rejected"}:
        problems.append(f"invalid status {report['status']!r}")
    verdict = report["verdict"]
    if not isinstance(verdict, dict) or verdict.get("status") not in {
        "pass",
        "fail",
        "not_evaluated",
    }:
        problems.append("verdict.status must be pass, fail, or not_evaluated")
    if not isinstance(report["warnings"], list):
        problems.append("warnings must be a list")
    else:
        for index, warning in enumerate(report["warnings"]):
            if (
                not isinstance(warning, dict)
                or not {
                    "code",
                    "severity",
                    "message",
                    "context",
                }
                <= warning.keys()
            ):
                problems.append(f"warnings[{index}] has the wrong shape")
    if not isinstance(report["metrics"], dict):
        problems.append("metrics must be an object")
    else:
        for name, metric in report["metrics"].items():
            if (
                not isinstance(metric, dict)
                or not {
                    "value",
                    "unit",
                    "status",
                }
                <= metric.keys()
            ):
                problems.append(f"metric {name!r} has the wrong shape")
            elif metric["status"] not in {
                "measured",
                "not_measured",
                "not_applicable",
            }:
                problems.append(f"metric {name!r} has invalid status")
    if not isinstance(report["timings_ms"], dict):
        problems.append("timings_ms must be an object")
    else:
        for name, value in report["timings_ms"].items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                problems.append(f"timing {name!r} must be a finite non-negative number")
    reproducibility = report.get("reproducibility")
    if not isinstance(reproducibility, dict):
        problems.append("reproducibility must be an object")
    elif reproducibility.get("scope") != "same_runtime_toolchain":
        problems.append("reproducibility.scope must be same_runtime_toolchain")
    return problems
