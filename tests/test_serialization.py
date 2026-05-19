# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
import json
import struct

import pytest

from chitin import Config, extract_from_mesh, extract_from_rigged_mesh


def test_json_round_trip(box_mesh, tmp_path):
    verts, faces = box_mesh
    r = extract_from_mesh(verts, faces, config=Config(concavity=0.5))
    out = tmp_path / "out.json"
    r.to_json(out)

    data = json.loads(out.read_text())
    assert len(data["hulls"]) == len(r.hulls)
    assert data["meta"]["hull_count"] == len(r.hulls)
    assert data["meta"]["source_vertex_count"] == r.source_vertex_count
    assert "bones" not in data["meta"]


def test_json_rigged(two_bone_rig, tmp_path):
    r = extract_from_rigged_mesh(**two_bone_rig, config=Config(concavity=0.5))
    out = tmp_path / "rigged.json"
    r.to_json(out)

    data = json.loads(out.read_text())
    assert data["meta"]["rigged"] is True
    assert len(data["meta"]["bones"]) == 2
    for bone in data["meta"]["bones"]:
        assert "bind_transform" in bone
        assert len(bone["bind_transform"]) == 4
        assert len(bone["bind_transform"][0]) == 4
    for hull in data["hulls"]:
        assert "bone_name" in hull
        assert "bone_index" in hull


def test_phys_header(box_mesh, tmp_path):
    verts, faces = box_mesh
    r = extract_from_mesh(verts, faces, config=Config(concavity=0.5))
    out = tmp_path / "out.phys"
    r.to_phys(out)

    with open(out, "rb") as f:
        magic = f.read(4)
        assert magic == b"PHYS"
        version = struct.unpack("<H", f.read(2))[0]
        assert version == 3
        flags = struct.unpack("<H", f.read(2))[0]
        assert flags == 0
        hull_count = struct.unpack("<I", f.read(4))[0]
        assert hull_count == len(r.hulls)


def test_phys_rigged_flags(two_bone_rig, tmp_path):
    r = extract_from_rigged_mesh(**two_bone_rig, config=Config(concavity=0.5))
    out = tmp_path / "rigged.phys"
    r.to_phys(out)

    with open(out, "rb") as f:
        f.read(4)
        f.read(2)
        flags = struct.unpack("<H", f.read(2))[0]
        assert flags & 0x01, "FLAG_HAS_BONES"
        assert flags & 0x02, "FLAG_HAS_BIND_POSES"


def test_usd_output(box_mesh, tmp_path):
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError:
        pytest.skip("usd-core not installed")

    verts, faces = box_mesh
    r = extract_from_mesh(verts, faces, config=Config(concavity=0.5))
    out = tmp_path / "out.usda"
    r.to_usd(out)

    stage = Usd.Stage.Open(str(out))
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert len(meshes) == len(r.hulls)
    for m in meshes:
        assert UsdPhysics.CollisionAPI(m)


def test_usd_rigged_bone_xforms(two_bone_rig, tmp_path):
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        pytest.skip("usd-core not installed")

    r = extract_from_rigged_mesh(**two_bone_rig, config=Config(concavity=0.5))
    out = tmp_path / "rigged.usda"
    r.to_usd(out)

    stage = Usd.Stage.Open(str(out))
    bone_prims = [
        p for p in stage.Traverse() if p.GetName() in ("left_arm", "right_arm")
    ]
    assert len(bone_prims) == 2
    for prim in bone_prims:
        xformable = UsdGeom.Xformable(prim)
        assert len(xformable.GetOrderedXformOps()) > 0
