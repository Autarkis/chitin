# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
import struct

import numpy as np

from chitin import (
    Config,
    extract_from_mesh,
    extract_from_rigged_mesh,
    read_phys,
    validate_phys,
)
from chitin.phys import FLAG_HAS_BIND_POSES, FLAG_HAS_BONES, HEADER_SIZE


def test_round_trip_static(box_mesh, tmp_path):
    verts, faces = box_mesh
    result = extract_from_mesh(verts, faces, config=Config(concavity=0.5))
    path = tmp_path / "static.phys"
    result.to_phys(path)

    pf = read_phys(path)
    assert pf.version == 2
    assert len(pf.hulls) == len(result.hulls)
    assert not pf.has_bones
    assert not pf.has_bind_poses
    assert pf.bones == []

    for orig, loaded in zip(result.hulls, pf.hulls):
        np.testing.assert_allclose(
            loaded.vertices,
            orig.vertices,
            atol=0.01,
            err_msg="dequantized vertices should approximate originals",
        )
        np.testing.assert_array_equal(loaded.indices, orig.indices)


def test_round_trip_rigged(two_bone_rig, tmp_path):
    result = extract_from_rigged_mesh(**two_bone_rig, config=Config(concavity=0.5))
    path = tmp_path / "rigged.phys"
    result.to_phys(path)

    pf = read_phys(path)
    assert pf.has_bones
    assert pf.has_bind_poses
    assert len(pf.bones) == 2
    assert {b.name for b in pf.bones} == {"left_arm", "right_arm"}

    for bone_orig, bone_loaded in zip(result.bones, pf.bones):
        np.testing.assert_allclose(
            bone_loaded.bind_transform,
            bone_orig.bind_transform.astype(np.float32),
            atol=1e-6,
        )

    for h in pf.hulls:
        assert h.bone_index is not None


def test_validate_clean(box_mesh, tmp_path):
    verts, faces = box_mesh
    result = extract_from_mesh(verts, faces, config=Config(concavity=0.5))
    path = tmp_path / "clean.phys"
    result.to_phys(path)

    issues = validate_phys(path)
    errors = [i for i in issues if i.severity == "error"]
    assert errors == [], f"unexpected errors: {errors}"


def test_validate_rigged_clean(two_bone_rig, tmp_path):
    result = extract_from_rigged_mesh(**two_bone_rig, config=Config(concavity=0.5))
    path = tmp_path / "rigged_clean.phys"
    result.to_phys(path)

    issues = validate_phys(path)
    errors = [i for i in issues if i.severity == "error"]
    assert errors == [], f"unexpected errors: {errors}"


def test_validate_bad_magic(tmp_path):
    path = tmp_path / "bad.phys"
    path.write_bytes(b"NOPE" + b"\x00" * 28)
    issues = validate_phys(path)
    assert any("bad magic" in i.message for i in issues)


def test_validate_truncated(tmp_path):
    path = tmp_path / "tiny.phys"
    path.write_bytes(b"PHYS\x02\x00")
    issues = validate_phys(path)
    assert any("too small" in i.message for i in issues)


def test_validate_bad_offsets(box_mesh, tmp_path):
    verts, faces = box_mesh
    result = extract_from_mesh(verts, faces, config=Config(concavity=0.5))
    path = tmp_path / "offsets.phys"
    result.to_phys(path)

    data = bytearray(path.read_bytes())
    struct.pack_into("<I", data, 20, 999)
    path.write_bytes(data)

    issues = validate_phys(path)
    assert any("hull_table_offset" in i.message for i in issues)


def test_header_layout(box_mesh, tmp_path):
    verts, faces = box_mesh
    result = extract_from_mesh(verts, faces, config=Config(concavity=0.5))
    path = tmp_path / "layout.phys"
    result.to_phys(path)

    data = path.read_bytes()
    assert data[:4] == b"PHYS"
    version, flags = struct.unpack_from("<HH", data, 4)
    assert version == 2
    assert flags == 0

    hull_count, total_verts, total_idx = struct.unpack_from("<III", data, 8)
    hull_table_off, vertex_data_off, index_data_off = struct.unpack_from(
        "<III", data, 20
    )

    assert hull_table_off == HEADER_SIZE
    assert vertex_data_off == hull_table_off + hull_count * 40
    assert index_data_off == vertex_data_off + total_verts * 6
    assert len(data) == index_data_off + total_idx * 2


def test_rigged_flags_layout(two_bone_rig, tmp_path):
    result = extract_from_rigged_mesh(**two_bone_rig, config=Config(concavity=0.5))
    path = tmp_path / "flags.phys"
    result.to_phys(path)

    data = path.read_bytes()
    flags = struct.unpack_from("<H", data, 6)[0]
    assert flags & FLAG_HAS_BONES
    assert flags & FLAG_HAS_BIND_POSES

    hull_count = struct.unpack_from("<I", data, 8)[0]
    hull_table_off = struct.unpack_from("<I", data, 20)[0]
    vertex_data_off = struct.unpack_from("<I", data, 24)[0]
    assert vertex_data_off == hull_table_off + hull_count * 44


def test_validate_bind_pose_flag_without_block(box_mesh, tmp_path):
    verts, faces = box_mesh
    result = extract_from_mesh(verts, faces, config=Config(concavity=0.5))
    path = tmp_path / "fake_bones.phys"
    result.to_phys(path)

    data = bytearray(path.read_bytes())
    struct.pack_into("<H", data, 6, 0x03)
    path.write_bytes(data)

    issues = validate_phys(path)
    assert any(
        "HAS_BIND_POSES" in i.message or "bone" in i.message.lower() for i in issues
    )


def test_quantization_precision(box_mesh, tmp_path):
    verts, faces = box_mesh
    result = extract_from_mesh(verts, faces, config=Config(concavity=0.5))
    path = tmp_path / "quant.phys"
    result.to_phys(path)

    pf = read_phys(path)
    for orig, loaded in zip(result.hulls, pf.hulls):
        extent = orig.vertices.max(axis=0) - orig.vertices.min(axis=0)
        max_extent = max(extent)
        if max_extent > 0:
            max_error = max_extent / 65535
            np.testing.assert_allclose(
                loaded.vertices,
                orig.vertices,
                atol=max_error * 1.5,
            )
