import numpy as np

from chitin import Config, extract_from_rigged_mesh


def test_per_bone_hulls(two_bone_rig):
    r = extract_from_rigged_mesh(
        **two_bone_rig,
        config=Config(concavity=0.5),
    )
    assert len(r.hulls) == 2
    bone_names = {h.bone_name for h in r.hulls}
    assert bone_names == {"left_arm", "right_arm"}


def test_hulls_in_bone_local_space(two_bone_rig):
    r = extract_from_rigged_mesh(
        **two_bone_rig,
        config=Config(concavity=0.5),
    )
    for hull in r.hulls:
        center = hull.vertices.mean(axis=0)
        assert abs(center[0]) < 1.0, f"{hull.bone_name} not centered: {center}"


def test_bind_transform_reconstructs_world(two_bone_rig):
    r = extract_from_rigged_mesh(
        **two_bone_rig,
        config=Config(concavity=0.5),
    )
    for hull in r.hulls:
        bind = r.bones[hull.bone_index].bind_transform
        local_pts = hull.vertices.astype(np.float64)
        ones = np.ones((len(local_pts), 1), dtype=np.float64)
        world_pts = (np.hstack([local_pts, ones]) @ bind)[:, :3]
        world_center = world_pts.mean(axis=0)
        expected_x = -1.0 if hull.bone_name == "left_arm" else 1.0
        assert abs(world_center[0] - expected_x) < 1.0


def test_bones_metadata(two_bone_rig):
    r = extract_from_rigged_mesh(
        **two_bone_rig,
        config=Config(concavity=0.5),
    )
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
