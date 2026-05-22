from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


COMPONENT_TYPE_SIZE = {
    5120: 1,  # BYTE
    5121: 1,  # UNSIGNED_BYTE
    5122: 2,  # SHORT
    5123: 2,  # UNSIGNED_SHORT
    5125: 4,  # UNSIGNED_INT
    5126: 4,  # FLOAT
}
COMPONENT_TYPE_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


@dataclass
class GltfSkinData:
    joint_names: list[str]
    inverse_bind_matrices: dict[int, np.ndarray]
    joint_indices: np.ndarray | None = None
    joint_weights: np.ndarray | None = None


def parse_skin(path: Path) -> GltfSkinData | None:
    suffix = path.suffix.lower()
    if suffix == ".glb":
        gltf_json, bin_data = _read_glb(path)
    elif suffix == ".gltf":
        gltf_json = json.loads(path.read_text())
        bin_path = gltf_json.get("buffers", [{}])[0].get("uri")
        bin_data = (path.parent / bin_path).read_bytes() if bin_path else b""
    else:
        return None

    skins = gltf_json.get("skins")
    if not skins:
        return None

    skin = skins[0]
    joint_node_indices = skin.get("joints", [])
    if not joint_node_indices:
        return None

    nodes = gltf_json.get("nodes", [])
    joint_names = []
    for idx in joint_node_indices:
        node = nodes[idx] if idx < len(nodes) else {}
        joint_names.append(node.get("name", f"joint_{idx}"))

    ibm_accessor_idx = skin.get("inverseBindMatrices")
    inverse_bind_matrices = {}
    if ibm_accessor_idx is not None:
        matrices = _read_accessor_mat4(gltf_json, bin_data, ibm_accessor_idx)
        for i, mat in enumerate(matrices):
            inverse_bind_matrices[i] = mat

    joint_indices = None
    joint_weights = None
    meshes = gltf_json.get("meshes", [])
    if meshes:
        prim = meshes[0].get("primitives", [{}])[0]
        attrs = prim.get("attributes", {})
        j_idx = attrs.get("JOINTS_0")
        w_idx = attrs.get("WEIGHTS_0")
        if j_idx is not None and w_idx is not None:
            joint_indices = _read_accessor(gltf_json, bin_data, j_idx)
            joint_weights = _read_accessor(gltf_json, bin_data, w_idx)

    return GltfSkinData(
        joint_names=joint_names,
        inverse_bind_matrices=inverse_bind_matrices,
        joint_indices=joint_indices,
        joint_weights=joint_weights,
    )


def _read_glb(path: Path) -> tuple[dict, bytes]:
    with open(path, "rb") as f:
        magic, version, length = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67:
            raise ValueError("not a valid GLB file")

        json_chunk_len, json_chunk_type = struct.unpack("<II", f.read(8))
        json_data = json.loads(f.read(json_chunk_len))

        bin_data = b""
        if f.tell() < length:
            bin_chunk_len, bin_chunk_type = struct.unpack("<II", f.read(8))
            bin_data = f.read(bin_chunk_len)

    return json_data, bin_data


def _read_accessor(gltf: dict, bin_data: bytes, accessor_idx: int) -> np.ndarray:
    accessor = gltf["accessors"][accessor_idx]
    bv_idx = accessor.get("bufferView")
    count = accessor["count"]
    comp_type = accessor["componentType"]
    acc_type = accessor["type"]
    acc_offset = accessor.get("byteOffset", 0)

    dtype = COMPONENT_TYPE_DTYPE[comp_type]
    n_components = TYPE_COUNT[acc_type]
    comp_size = COMPONENT_TYPE_SIZE[comp_type]

    if bv_idx is None:
        return np.zeros((count, n_components), dtype=dtype)

    bv = gltf["bufferViews"][bv_idx]
    bv_offset = bv.get("byteOffset", 0)
    byte_stride = bv.get("byteStride", 0)
    start = bv_offset + acc_offset
    tight_stride = comp_size * n_components

    if byte_stride and byte_stride != tight_stride:
        if start + (count - 1) * byte_stride + tight_stride > len(bin_data):
            return np.zeros((count, n_components), dtype=dtype)

        if byte_stride % comp_size == 0:
            arr = np.ndarray(
                shape=(count, n_components),
                dtype=dtype,
                buffer=bin_data,
                offset=start,
                strides=(byte_stride, comp_size),
            )
            return arr.copy()

        # Fallback for unaligned byte strides (rare)
        buf = np.frombuffer(bin_data, dtype=np.uint8)
        out = np.empty((count, n_components), dtype=dtype)
        for i in range(count):
            elem_start = start + i * byte_stride
            elem_bytes = buf[elem_start : elem_start + tight_stride]
            out[i] = np.frombuffer(elem_bytes.tobytes(), dtype=dtype)
        return out

    total = count * n_components
    arr = np.frombuffer(bin_data, dtype=dtype, count=total, offset=start)
    if n_components > 1:
        arr = arr.reshape(count, n_components)
    return arr.copy()


def _read_accessor_mat4(
    gltf: dict, bin_data: bytes, accessor_idx: int
) -> list[np.ndarray]:
    accessor = gltf["accessors"][accessor_idx]
    bv_idx = accessor.get("bufferView")
    count = accessor["count"]
    acc_offset = accessor.get("byteOffset", 0)

    if bv_idx is None:
        return [np.eye(4, dtype=np.float32) for _ in range(count)]

    bv = gltf["bufferViews"][bv_idx]
    bv_offset = bv.get("byteOffset", 0)
    byte_stride = bv.get("byteStride", 0)

    start = bv_offset + acc_offset
    mat_size = 16 * 4
    stride = byte_stride if byte_stride else mat_size

    matrices = []

    if stride % 4 == 0:
        total_bytes_needed = (count - 1) * stride + mat_size
        if start + total_bytes_needed <= len(bin_data):
            arr = np.ndarray(
                shape=(count, 4, 4),
                dtype=np.float32,
                buffer=bin_data,
                offset=start,
                strides=(stride, 16, 4),
            )
            return [mat.astype(np.float64) for mat in arr]

    # Fallback for unaligned or truncated
    for i in range(count):
        offset = start + i * stride
        raw = bin_data[offset : offset + mat_size]
        if len(raw) < mat_size:
            matrices.append(np.eye(4, dtype=np.float64))
            continue
        values = struct.unpack("<16f", raw)
        mat = np.array(values, dtype=np.float64).reshape(4, 4)
        matrices.append(mat)

    return matrices
