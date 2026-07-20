import json

import pytest
import trimesh

import chitin
from chitin.acceptance import apply_profile, evaluate, get_profile, report_metrics
from chitin.config import Config
from chitin.exporters.bundle import export_bundle
from chitin.manifest import (
    MANIFEST_FILENAME,
    build_manifest,
    quality_warnings,
    verify_bundle,
    write_manifest,
)


@pytest.fixture
def box_input(tmp_path):
    path = tmp_path / "box.glb"
    trimesh.creation.box(extents=[2, 2, 2]).export(str(path), file_type="glb")
    return path


@pytest.fixture
def box_result(box_input):
    return chitin.extract(box_input, Config())


def test_bundle_carries_manifest(tmp_path, box_input, box_result):
    profile = get_profile("robotics")
    verdict = evaluate(profile.policy, report_metrics(box_result))
    bundle = tmp_path / "out"
    export_bundle(
        box_result,
        bundle,
        fmt="phys",
        input_path=box_input,
        config=Config(),
        verdict=verdict,
    )

    manifest = json.loads((bundle / MANIFEST_FILENAME).read_text())
    assert manifest["manifest_version"] == 1
    assert manifest["phys_version"] == 3
    assert manifest["compiler_version"].startswith(chitin.__version__)
    assert "coacd" in manifest["dependency_versions"]
    assert manifest["input"]["sha256"]
    assert manifest["config"]["hash"]
    assert manifest["quality"]["verdict"]["profile"] == "robotics"

    # Every emitted file (scene.phys + the json sidecars) is hashed.
    names = {o["file"] for o in manifest["outputs"]}
    assert "scene.phys" in names
    assert MANIFEST_FILENAME not in names  # it never hashes itself


def test_verify_bundle_clean_then_tampered(tmp_path, box_input, box_result):
    bundle = tmp_path / "out"
    export_bundle(box_result, bundle, fmt="phys", input_path=box_input, config=Config())

    assert verify_bundle(bundle) == []

    target = bundle / "scene.phys"
    target.write_bytes(target.read_bytes() + b"tampered")
    problems = verify_bundle(bundle)
    assert len(problems) == 1
    assert "scene.phys" in problems[0]
    assert "mismatch" in problems[0]


def test_verify_bundle_missing_output(tmp_path, box_input, box_result):
    bundle = tmp_path / "out"
    export_bundle(box_result, bundle, fmt="phys", input_path=box_input, config=Config())
    (bundle / "scene.phys").unlink()
    problems = verify_bundle(bundle)
    assert any("missing" in p for p in problems)


def test_verify_bundle_no_manifest(tmp_path):
    assert verify_bundle(tmp_path) == [f"missing {MANIFEST_FILENAME}"]


def test_manifest_without_verdict_is_plain_summary(tmp_path, box_result):
    # Before acceptance is computed, the manifest still carries a quality
    # summary (metrics), just no verdict.
    write_manifest(
        tmp_path,
        output_files=[],
        metrics=report_metrics(box_result),
    )
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    assert "verdict" not in manifest["quality"]
    assert manifest["quality"]["metrics"]["hull_count"] >= 1


def test_config_hash_is_stable_and_config_sensitive(tmp_path):
    m1 = build_manifest(
        output_dir=tmp_path, output_files=[], config_dict={"concavity": 0.05}
    )
    m2 = build_manifest(
        output_dir=tmp_path, output_files=[], config_dict={"concavity": 0.05}
    )
    m3 = build_manifest(
        output_dir=tmp_path, output_files=[], config_dict={"concavity": 0.01}
    )
    assert m1["config"]["hash"] == m2["config"]["hash"]
    assert m1["config"]["hash"] != m3["config"]["hash"]


def test_quality_warnings_flags_fallback(box_result):
    # No warnings on a clean box build...
    assert quality_warnings(box_result) == []
    # ...but a tagged fallback surfaces.
    box_result.build_plan.detected["fallback_hulls"] = 2
    warnings = quality_warnings(box_result)
    assert any("fallback" in w for w in warnings)


def test_profiles_produce_different_configs():
    # Two profiles must not produce byte-identical builds: robotics decomposes
    # tighter than interactive, so the resolved config (and thus geometry) differs.
    inter_cfg = apply_profile(Config(), get_profile("interactive"))
    rob_cfg = apply_profile(Config(), get_profile("robotics"))
    assert inter_cfg.concavity != rob_cfg.concavity
