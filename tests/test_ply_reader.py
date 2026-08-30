"""Tests for the permissive PLY reader that replaced GPL-licensed plyfile."""

from __future__ import annotations

import struct

import numpy as np

from chitin.adapters.ply import load_ply
from chitin.adapters.ply_reader import read_ply_mesh, read_ply_vertex
from chitin.analyze import analyze_input
from chitin.config import Config
from chitin.resolve import resolve_config

_SPLAT_PROPS = [
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
]

_SPLAT_VERTS = [
    (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, -3.0, -3.1, -3.2, 1.0, 0.0, 0.0, 0.0),
    (1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.1, -2.0, -2.1, -2.2, 0.7, 0.7, 0.0, 0.0),
]

_TETRA_VERTS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
]

_TETRA_FACES = [
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 3),
    (1, 2, 3),
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
        f.writelines(
            struct.pack(endian + "f" * len(_SPLAT_PROPS), *v) for v in _SPLAT_VERTS
        )


def _write_binary_mesh(path, vertices, faces, endian="<"):
    fmt = "binary_little_endian" if endian == "<" else "binary_big_endian"
    header = (
        "ply\n"
        f"format {fmt} 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.writelines(struct.pack(endian + "fff", *v) for v in vertices)
        for face in faces:
            f.write(struct.pack(endian + "B", len(face)))
            f.write(struct.pack(endian + "i" * len(face), *face))


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
        f.writelines(
            struct.pack("<fff", *v)
            for v in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        )
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


def test_read_ply_mesh_binary_little_endian(tmp_path):
    p = tmp_path / "tetra.ply"
    _write_binary_mesh(p, _TETRA_VERTS, _TETRA_FACES, "<")
    ve, faces = read_ply_mesh(p)
    assert len(ve) == 4
    assert faces is not None
    assert faces.dtype == np.int32
    assert faces.shape == (4, 3)
    assert [tuple(int(x) for x in row) for row in faces] == _TETRA_FACES


def test_read_ply_mesh_binary_big_endian(tmp_path):
    p = tmp_path / "tetra_be.ply"
    _write_binary_mesh(p, _TETRA_VERTS, _TETRA_FACES, ">")
    ve, faces = read_ply_mesh(p)
    assert len(ve) == 4
    assert faces.shape == (4, 3)
    assert [tuple(int(x) for x in row) for row in faces] == _TETRA_FACES


def test_read_ply_mesh_ascii(tmp_path):
    p = tmp_path / "tetra_ascii.ply"
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(_TETRA_VERTS)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(_TETRA_FACES)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    body = "".join(f"{x} {y} {z}\n" for x, y, z in _TETRA_VERTS)
    body += "".join(f"3 {a} {b} {c}\n" for a, b, c in _TETRA_FACES)
    p.write_text(header + body)
    ve, faces = read_ply_mesh(p)
    assert len(ve) == 4
    assert faces.shape == (4, 3)
    assert [tuple(int(x) for x in row) for row in faces] == _TETRA_FACES


def test_read_ply_mesh_quad_fan_triangulates(tmp_path):
    p = tmp_path / "quad.ply"
    verts = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    _write_binary_mesh(p, verts, [(0, 1, 2, 3)], "<")
    _, faces = read_ply_mesh(p)
    assert faces.shape == (2, 3)
    assert tuple(int(x) for x in faces[0]) == (0, 1, 2)
    assert tuple(int(x) for x in faces[1]) == (0, 2, 3)


def test_read_ply_mesh_no_face_element_returns_none(tmp_path):
    p = tmp_path / "splat.ply"
    _write_binary_splat(p, "<")
    ve, faces = read_ply_mesh(p)
    assert len(ve) == 2
    assert faces is None


def test_read_ply_mesh_zero_faces_returns_none(tmp_path):
    p = tmp_path / "empty_faces.ply"
    _write_binary_mesh(p, _TETRA_VERTS, [], "<")
    _, faces = read_ply_mesh(p)
    assert faces is None


def test_read_ply_vertex_unaffected_by_face_element(tmp_path):
    p = tmp_path / "tetra.ply"
    _write_binary_mesh(p, _TETRA_VERTS, _TETRA_FACES, "<")
    ve = read_ply_vertex(p)
    assert len(ve) == 4
    assert ve.data.dtype.names == ("x", "y", "z")


def test_load_ply_populates_faces_for_mesh(tmp_path):
    p = tmp_path / "tetra.ply"
    _write_binary_mesh(p, _TETRA_VERTS, _TETRA_FACES, "<")
    res = load_ply(p)
    assert res.faces is not None
    assert res.faces.shape == (4, 3)


def test_load_ply_splat_has_no_faces(tmp_path):
    p = tmp_path / "splat.ply"
    _write_binary_splat(p, "<")
    res = load_ply(p)
    assert res.faces is None


def test_analyze_input_reports_face_count_for_ply_mesh(tmp_path):
    p = tmp_path / "tetra.ply"
    _write_binary_mesh(p, _TETRA_VERTS, _TETRA_FACES, "<")
    analysis = analyze_input(p)
    assert analysis.face_count == 4


def test_analyze_input_face_count_none_for_ply_point_cloud(tmp_path):
    p = tmp_path / "splat.ply"
    _write_binary_splat(p, "<")
    analysis = analyze_input(p)
    assert analysis.face_count is None


def test_ply_mesh_gets_no_proximity_default(tmp_path):
    p = tmp_path / "tetra.ply"
    _write_binary_mesh(p, _TETRA_VERTS, _TETRA_FACES, "<")
    analysis = analyze_input(p)
    resolved = resolve_config(Config(), analysis)
    assert resolved.surface_proximity_filter == 0.0
    assert "surface_proximity_filter" not in resolved.decisions


def test_ply_point_cloud_still_gets_proximity_default(tmp_path):
    p = tmp_path / "splat.ply"
    _write_binary_splat(p, "<")
    analysis = analyze_input(p)
    resolved = resolve_config(Config(), analysis)
    assert resolved.surface_proximity_filter == 5.0
