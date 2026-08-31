import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

import chitin.report as report_module
from chitin._metric_names import (
    CLEARANCE_BLOCKED_FRACTION,
    DEEP_FALSE_FILL_FRACTION,
    FALLBACK_RATIO,
    FALSE_FILL_FRACTION,
    HULL_TRIANGLE_COUNT,
    HULL_VERTEX_COUNT,
    PLANAR_SUBSTITUTE_HULLS,
    RADIUS_BLOCKED_FRACTION,
    SEAM_SNAG_COUNT,
    SOURCE_SURFACE_COVERAGE,
    STANDABLE_FRACTION,
    SWEEP_TRAVERSABILITY,
)
from chitin.acceptance import Check, Verdict, evaluate, get_profile, report_metrics
from chitin.plan import BuildPlan
from chitin.report import (
    REPORT_FIELDS,
    REPORT_VERSION,
    build_compilation_report,
    validate_compilation_report,
)
from chitin.result import ExtractionResult, Hull

SCHEMA_PATH = Path(__file__).parents[1] / "docs" / "compilation-report.schema.json"


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _result(*, fallback_hulls=0):
    hull = Hull(
        vertices=np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=np.float32,
        ),
        indices=np.array([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3], dtype=np.uint32),
    )
    plan = BuildPlan(
        input_kind="glb",
        collider_kind="static",
        pipeline=["parse", "decompose", "coverage"],
        source_vertices=4,
        processed_vertices=4,
        detected={
            "coverage": {
                SOURCE_SURFACE_COVERAGE: 0.99,
                "worst_component_surface_coverage": None,
                "worst_decile_surface_coverage": None,
                FALSE_FILL_FRACTION: 0.01,
                DEEP_FALSE_FILL_FRACTION: 0.0,
            },
            "fallback_hulls": fallback_hulls,
            "coacd_timeouts": fallback_hulls,
            "coacd_deterministic": True,
            "snugfit_refined": 1,
        },
    )
    return ExtractionResult(
        hulls=[hull],
        source_vertex_count=4,
        mesh_vertex_count=4,
        build_plan=plan,
    )


def test_python_report_matches_v1_contract():
    result = _result()
    verdict = evaluate(get_profile("robotics").policy, report_metrics(result))
    report = build_compilation_report(
        result,
        profile="robotics",
        verdict=verdict,
        requested_config={"snug_fit": False},
        effective_config={"snug_fit": True},
        artifacts={"phys": "scene.phys"},
        artifact_bytes=128,
        artifact_sha256="a" * 64,
    ).to_dict()

    assert report["report_version"] == REPORT_VERSION
    assert report["verdict"]["status"] == "pass"
    assert report["output"] == {
        "collider_kind": "static",
        "hull_count": 1,
        "vertex_count": 4,
        "triangle_count": 4,
        "lod_tier_count": 0,
        "byte_length": 128,
    }
    assert report["metrics"][SOURCE_SURFACE_COVERAGE] == {
        "value": 0.99,
        "unit": "ratio",
        "status": "measured",
    }
    assert report["metrics"][HULL_VERTEX_COUNT]["value"] == 4
    assert report["metrics"][HULL_TRIANGLE_COUNT]["value"] == 4
    assert report["processing"]["refinements"]["snug_fit"]["status"] == "applied"
    assert report["reproducibility"] == {
        "scope": "same_runtime_toolchain",
        "deterministic": True,
        "artifact_sha256": "a" * 64,
    }
    assert validate_compilation_report(report) == []


def test_unevaluated_report_does_not_imply_pass():
    report = build_compilation_report(_result(), profile="walkable").to_dict()
    assert report["status"] == "complete"
    assert report["verdict"] == {
        "profile": "walkable",
        "status": "not_evaluated",
        "reasons": [],
        "checks": [],
    }


def test_requested_snug_fit_without_execution_stats_is_skipped():
    result = _result()
    result.build_plan.detected.pop("snugfit_refined")

    report = build_compilation_report(
        result,
        effective_config={"snug_fit": True},
    ).to_dict()

    assert report["processing"]["refinements"]["snug_fit"]["status"] == "skipped"


def test_fallback_warning_is_typed_and_separate_from_planar_substitute():
    report = build_compilation_report(_result(fallback_hulls=1)).to_dict()
    assert report["warnings"][0]["code"] == "COACD_TIMEOUT_FALLBACK"
    assert report["warnings"][0]["context"] == {"hull_count": 1}
    assert report["processing"]["fallbacks"] == {
        "decomposition_failure_hulls": 1,
        "planar_substitute_hulls": 0,
    }
    assert report["metrics"][FALLBACK_RATIO]["value"] == 1.0
    assert report["metrics"][PLANAR_SUBSTITUTE_HULLS]["value"] == 0


def test_walkable_sweep_metrics_are_present_in_canonical_report():
    result = _result()
    result.build_plan.detected["sweep"] = {
        "traversability": 0.75,
        "standable_fraction": 0.80,
        "clearance_blocked_fraction": 0.10,
        "radius_blocked_fraction": 0.20,
        "seam_snags": [(0.0, 0.2, 0.0), (1.0, 0.4, 1.0)],
    }

    metrics = build_compilation_report(result, profile="walkable").to_dict()["metrics"]

    assert metrics[SWEEP_TRAVERSABILITY]["value"] == 0.75
    assert metrics[STANDABLE_FRACTION]["value"] == 0.80
    assert metrics[CLEARANCE_BLOCKED_FRACTION]["value"] == 0.10
    assert metrics[RADIUS_BLOCKED_FRACTION]["value"] == 0.20
    assert metrics[SEAM_SNAG_COUNT]["value"] == 2


def test_schema_and_python_require_the_same_top_level_fields():
    schema = _schema()
    assert tuple(schema["required"]) == REPORT_FIELDS


def test_validator_rejects_unknown_version_and_missing_shape():
    problems = validate_compilation_report({"report_version": 99})
    assert any("is a required property" in problem for problem in problems)


def test_schema_file_is_a_valid_json_schema():
    jsonschema.Draft202012Validator.check_schema(_schema())


def test_schema_loader_falls_back_to_source_checkout(monkeypatch):
    class MissingPackageResource:
        def read_text(self, *, encoding):
            raise FileNotFoundError

    report_module._load_schema.cache_clear()
    monkeypatch.setattr(report_module, "_PACKAGED_SCHEMA", MissingPackageResource())
    assert report_module._load_schema()["$schema"].endswith("2020-12/schema")
    report_module._load_schema.cache_clear()


def test_complete_report_validates_against_schema_file():
    # Same producer path as test_python_report_matches_v1_contract: a real
    # evaluate() verdict that passes, so status == "complete".
    result = _result()
    verdict = evaluate(get_profile("robotics").policy, report_metrics(result))
    report = build_compilation_report(
        result,
        profile="robotics",
        verdict=verdict,
        requested_config={"snug_fit": False},
        effective_config={"snug_fit": True},
        artifacts={"phys": "scene.phys"},
        artifact_bytes=128,
        artifact_sha256="a" * 64,
    ).to_dict()

    assert report["status"] == "complete"
    jsonschema.validate(instance=report, schema=_schema())
    assert validate_compilation_report(report) == []


def test_rejected_report_validates_against_schema_file():
    # Same real producer path, but with a failing Verdict (as
    # test_service.py's test_failed_acceptance_rejects_job forces), so
    # status == "rejected" exercises that branch.
    result = _result()
    verdict = Verdict(
        profile="robotics",
        passed=False,
        checks=(Check("forced", False, "forced failure for test"),),
    )
    report = build_compilation_report(
        result,
        profile="robotics",
        verdict=verdict,
        artifacts={"phys": "scene.phys"},
        artifact_bytes=128,
        artifact_sha256="a" * 64,
    ).to_dict()

    assert report["status"] == "rejected"
    jsonschema.validate(instance=report, schema=_schema())
    assert validate_compilation_report(report) == []


def test_missing_fields_rejection_is_also_a_schema_violation():
    bad_report = {"report_version": 99}
    assert validate_compilation_report(bad_report) != []
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad_report, schema=_schema())


def test_bad_status_rejection_is_also_a_schema_violation():
    # Corrupt a single field on an otherwise-real, schema-valid report so
    # this exercises the enum check specifically, not the missing-fields
    # short-circuit that {"report_version": 99} alone would hit.
    result = _result()
    verdict = evaluate(get_profile("robotics").policy, report_metrics(result))
    report = build_compilation_report(
        result, profile="robotics", verdict=verdict
    ).to_dict()
    report["status"] = "bogus"

    assert validate_compilation_report(report) != []
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=report, schema=_schema())


def test_validator_rejects_boolean_timing():
    report = build_compilation_report(_result()).to_dict()
    report["timings_ms"] = {"total": True}

    assert validate_compilation_report(report) == [
        "timings_ms.total: True is not of type 'number'"
    ]


def test_validator_reports_invalid_reproducibility_container():
    report = build_compilation_report(_result()).to_dict()
    report["reproducibility"] = []

    assert validate_compilation_report(report) == [
        "reproducibility: [] is not of type 'object'"
    ]


def test_validator_falls_back_to_hand_rolled_checks_without_jsonschema(monkeypatch):
    # Simulates a minimal runtime install where jsonschema isn't installed:
    # validate_compilation_report() must still catch structural problems,
    # just via the hand-rolled fallback (with its own message shape).
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jsonschema":
            raise ImportError("simulated missing optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    missing_problems = validate_compilation_report({"report_version": 99})
    assert missing_problems == [
        (
            "missing fields: status, profile, verdict, input, output, timings_ms, "
            "warnings, metrics, processing, runtime, reproducibility, config, artifacts"
        )
    ]

    report = build_compilation_report(_result()).to_dict()
    report["timings_ms"] = {"total": True}
    assert validate_compilation_report(report) == [
        "timing 'total' must be a finite non-negative number"
    ]
