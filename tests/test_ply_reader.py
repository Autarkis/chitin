"""Tests for the permissive PLY reader that replaced GPL-licensed plyfile."""

from __future__ import annotations

import struct

import numpy as np

from chitin.adapters.ply import load_ply
from chitin.adapters.ply_reader import read_ply_vertex

_SPLAT_PROPS = [
    "x", "y", "z",
    "nx", "ny", "nz",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]

_SPLAT_VERTS = [
    (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, -3.0, -3.1, -3.2, 1.0, 0.0, 0.0, 0.0),
    (1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.1, -2.0, -2.1, -2.2, 0.7, 0.7, 0.0, 0.0),
]


def _write_binary_splat(path, endian="<"):
    fmt = "binary_little_endian" if endian == "<" else "binary_big_endian"
    header = (
        "ply\n"
        f"format {fmt} 1.0\n"
        f"element vertex {len(_SPLAT_VERTS)}\n"
        + "".join(f"property float {p}\n" for p in _SPLAT_PROPS)
        + "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        for v in _SPLAT_VERTS:
            f.write(struct.pack(endian + "f" * len(_SPLAT_PROPS), *v))


def test_binary_little_endian(tmp_path):
    p = tmp_path / "splat.ply"
    _write_binary_splat(p, "<")
    ve = read_ply_vertex(p)
    assert len(ve) == 2
    assert ve.data.dtype.names == tuple(_SPLAT_PROPS)
    assert ve["x"][1] == 1.0 and ve["z"][1] == 3.0
    assert abs(float(ve["opacity"][0]) - 0.9) < 1e-6
    assert ve["scale_2"][0] == np.float32(-3.2)


def test_binary_big_endian(tmp_path):
    p = tmp_path / "splat_be.ply"
    _write_binary_splat(p, ">")
    ve = read_ply_vertex(p)
    assert ve["y"][1] == 2.0 and ve["rot_1"][1] == np.float32(0.7)


def test_ascii(tmp_path):
    p = tmp_path / "cloud.ply"
    header = (
        "ply\nformat ascii 1.0\nelement vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nend_header\n"
    )
    p.write_text(header + "0 0 0 255\n1.5 2.5 3.5 128\n-1 -2 -3 0\n")
    ve = read_ply_vertex(p)
    assert len(ve) == 3
    assert ve["x"][1] == 1.5 and ve["z"][2] == -3.0
    assert ve["red"][0] == 255 and ve["red"][2] == 0


def test_skips_trailing_face_element(tmp_path):
    # A vertex element followed by a face element with a list property: the
    # reader must parse past the faces and still return the 3 vertices.
    p = tmp_path / "mesh.ply"
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 1\nproperty list uchar int vertex_indices\n"
        "end_header\n"
    )
    with open(p, "wb") as f:
        f.write(header.encode())
        for v in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]:
            f.write(struct.pack("<fff", *v))
        f.write(struct.pack("<B", 3) + struct.pack("<iii", 0, 1, 2))
    ve = read_ply_vertex(p)
    assert len(ve) == 3
    assert ve["x"][1] == 1.0 and ve["y"][2] == 1.0


def test_load_ply_reads_covariance_splat(tmp_path):
    p = tmp_path / "splat.ply"
    _write_binary_splat(p, "<")
    res = load_ply(p)
    assert res.format == "ply"
    assert res.positions.shape == (2, 3)
    assert res.positions[1, 0] == 1.0
    assert res.detected["has_covariance"] is True
    assert res.opacity is not None
    assert res.scales.shape == (2, 3) and res.rots.shape == (2, 4)


def test_load_ply_reads_normals_cloud(tmp_path):
    # No scale/rot -> falls back to normals.
    p = tmp_path / "cloud.ply"
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "end_header\n"
    )
    with open(p, "wb") as f:
        f.write(header.encode())
        f.write(struct.pack("<ffffff", 0, 0, 0, 0, 0, 1))
        f.write(struct.pack("<ffffff", 1, 1, 1, 1, 0, 0))
    res = load_ply(p)
    assert res.detected["has_covariance"] is False
    assert res.normals is not None and res.normals.shape == (2, 3)
