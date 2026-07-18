"""FBX inputs auto-convert to GLB via Blender, transparently, in load()."""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

import chitin.convert as convertmod
from chitin.adapters import load
from chitin.convert import BlenderNotFoundError


def test_fbx_routes_through_blender_convert(monkeypatch, tmp_path):
    # A real GLB the fake converter "produces" from the .fbx input.
    glb_bytes = trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(file_type="glb")

    captured = {}

    def fake_convert(input_path, output_path):
        captured["input"] = Path(input_path)
        Path(output_path).write_bytes(glb_bytes)

    monkeypatch.setattr(convertmod, "convert_fbx_to_glb", fake_convert)

    fbx = tmp_path / "model.fbx"
    fbx.write_bytes(b"stub; never parsed, the converter is faked")

    result = load(fbx)

    assert captured["input"] == fbx
    assert result.faces is not None and len(result.faces) > 0
    assert result.detected.get("converted_from_fbx") is True


def test_fbx_without_blender_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(convertmod, "find_blender", lambda: None)

    fbx = tmp_path / "model.fbx"
    fbx.write_bytes(b"stub")

    with pytest.raises(BlenderNotFoundError):
        load(fbx)
