import numpy as np

from chitin import Config, extract_from_rigged_mesh
from chitin._metric_names import SOURCE_SURFACE_COVERAGE
from chitin.acceptance import evaluate, get_profile, report_metrics


def test_per_bone_hulls(rigged_result):
    r = rigged_result
    assert len(r.hulls) == 2
    bone_names = {h.bone_name for h in r.hulls}
    assert bone_names == {"left_arm", "right_arm"}


def test_hulls_in_bone_local_space(rigged_result):
    r = rigged_result
    for hull in r.hulls:
        center = hull.vertices.mean(axis=0)
        assert abs(center[0]) < 1.0, f"{hull.bone_name} not centered: {center}"


def test_bind_transform_reconstructs_world(rigged_result):
    r = rigged_result
    for hull in r.hulls:
        bind = r.bones[hull.bone_index].bind_transform
        local_pts = hull.vertices.astype(np.float64)
        ones = np.ones((len(local_pts), 1), dtype=np.float64)
        world_pts = (np.hstack([local_pts, ones]) @ bind)[:, :3]
        world_center = world_pts.mean(axis=0)
        expected_x = -1.0 if hull.bone_name == "left_arm" else 1.0
        assert abs(world_center[0] - expected_x) < 1.0


def test_bones_metadata(rigged_result):
    r = rigged_result
    assert r.bones is not None
    assert len(r.bones) == 2
    assert r.bones[0].name == "left_arm"
    assert r.bones[0].index == 0
    assert r.bones[1].name == "right_arm"
    assert r.bones[1].index == 1
    assert r.bones[0].bind_transform.shape == (4, 4)


def test_no_ibm_skips_transform(two_bone_rig):
    r = extract_from_rigged_mesh(
        vertices=two_bone_rig["vertices"],
        faces=two_bone_rig["faces"],
        joint_indices=two_bone_rig["joint_indices"],
        joint_weights=two_bone_rig["joint_weights"],
        bone_names=two_bone_rig["bone_names"],
        config=Config(concavity=0.5),
    )
    assert len(r.hulls) == 2
    for hull in r.hulls:
        center = hull.vertices.mean(axis=0)
        assert abs(center[0]) > 0.5, "should still be in world space"


def test_rigged_lod_tiers_are_populated(rigged_lod_result):
    # LOD tiers were computed per bone and then discarded; they must now survive
    # and carry hulls from every bone, tagged bone-local.
    r = rigged_lod_result
    assert r.lod_tiers is not None
    assert [round(t.concavity, 3) for t in r.lod_tiers] == [0.3, 0.7]
    for tier in r.lod_tiers:
        assert tier.hulls
        assert all(h.bone_name in {"left_arm", "right_arm"} for h in tier.hulls)


def test_rigged_build_reports_coverage(two_bone_rig):
    # The rigged path recorded no coverage at all, so `source_surface_coverage`
    # came back None and every strict profile rejected rigged assets outright.
    # The measurement is taken bind-posed, against the model-space input.
    r = extract_from_rigged_mesh(
        **two_bone_rig,
        config=Config(concavity=0.5, snug_fit=True),
    )
    coverage = r.build_plan.detected.get("coverage")
    assert coverage is not None
    assert coverage[SOURCE_SURFACE_COVERAGE] > 0.9

    metrics = report_metrics(r)
    assert metrics[SOURCE_SURFACE_COVERAGE] is not None
    verdict = evaluate(get_profile("robotics").policy, metrics)
    assert verdict.passed, verdict.checks


def test_rigged_merges_per_bone_fallback_counters(two_bone_rig, monkeypatch):
    # Bones decompose under their own plan, so a CoACD timeout inside one used
    # to die with it: the asset reported zero fallbacks and sailed through the
    # robotics gate with a bounding box in it.
    from chitin.stages import decompose

    def boom(*args, **kwargs):
        raise decompose.CoACDTimeoutError("forced")

    monkeypatch.setattr(decompose, "run_coacd_bounded", boom)
    r = extract_from_rigged_mesh(**two_bone_rig, config=Config(concavity=0.5))

    assert r.build_plan.detected["fallback_hulls"] == 2
    assert r.build_plan.detected["coacd_timeouts"] == 2
    # And the per-bone pipelines stay off the asset's step list.
    assert r.build_plan.pipeline.count("decompose") == 0

    verdict = evaluate(get_profile("robotics").policy, report_metrics(r))
    assert not verdict.passed
    assert any(c.name == "no_fallback_hulls" and not c.passed for c in verdict.checks)


def test_rigged_lod_roundtrips_through_phys(rigged_lod_result, tmp_path):
    r = rigged_lod_result
    out = tmp_path / "rigged_lod.phys"
    r.to_phys(out)

    from chitin.phys import read_phys, validate_phys

    assert validate_phys(out) == []
    pf = read_phys(out)
    assert pf.has_lod and pf.has_bones
    assert [round(t.concavity, 3) for t in pf.lod_tiers] == [0.3, 0.7]
