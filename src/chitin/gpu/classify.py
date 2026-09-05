"""GPU classify stage — Policy 0.3 vertex/plane classification.

Two-pass pipeline:
1. classify_norm: per-(mesh,plane) centroid, extent, grid-frame parameters
2. classify: per-vertex grid-frame classification with ambiguity fallback
"""

from __future__ import annotations

import math
import struct

import numpy as np

from chitin.gpu.layouts import CLIP_PLANE, MESH_HEADER, NORM_PARAMS, POINT
from chitin.gpu.shaders import load_shader
from chitin.gpu.worker import GPUWorker

_STORAGE = 0x80
_COPY_SRC = 0x04
_COPY_DST = 0x08
_UNIFORM = 0x40


def _buffer_from_numpy(worker: GPUWorker, array: np.ndarray) -> object:
    data = array.tobytes()
    buffer = worker.create_buffer(max(len(data), 4), _STORAGE | _COPY_DST)
    if data:
        worker.write_buffer(buffer, data)
    return buffer


def _rw_buffer(worker: GPUWorker, size: int) -> object:
    return worker.create_buffer(max(size, 4), _STORAGE | _COPY_SRC | _COPY_DST)


def _output_buffer(worker: GPUWorker, size: int) -> object:
    return worker.create_buffer(max(size, 4), _STORAGE | _COPY_SRC)


def _uniform_buffer(worker: GPUWorker, *values: int) -> object:
    data = struct.pack("<" + "I" * len(values), *values)
    padded = data + b"\x00" * (16 - len(data))
    buffer = worker.create_buffer(16, _UNIFORM | _COPY_DST)
    worker.write_buffer(buffer, padded)
    return buffer


def _dispatch_pass(
    worker: GPUWorker,
    shader_name: str,
    buffers: list[object],
    workgroups: tuple[int, int, int],
) -> None:
    operation = worker.begin_operation()
    pipeline = worker.create_compute_pipeline(load_shader(shader_name))
    bind_group = worker.create_bind_group(
        pipeline,
        0,
        [{"binding": i, "resource": {"buffer": buf}} for i, buf in enumerate(buffers)],
    )
    worker.dispatch_compute(pipeline, [bind_group], workgroups)
    worker.check_operation(operation)


def dispatch_classify(
    worker: GPUWorker,
    vertices: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    grid_bits: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify vertices against a clipping plane on GPU.

    Returns (signs, world_dots) where signs is i32 array of {-1, 0, +1}
    and world_dots is f32 array of world-frame signed distances.
    """
    vertices_f32 = np.asarray(vertices, dtype=np.float32)
    if vertices_f32.ndim != 2 or vertices_f32.shape[1] != 3:
        raise ValueError("vertices must be (N, 3)")
    n = len(vertices_f32)
    if n == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    plane_point_f32 = np.asarray(plane_point, dtype=np.float32).ravel()
    plane_normal_f32 = np.asarray(plane_normal, dtype=np.float32).ravel()

    # Pack vertices as Point structs (16B each: x, y, z, pad)
    verts_packed = np.zeros(n, dtype=POINT.dtype())
    verts_packed["x"] = vertices_f32[:, 0]
    verts_packed["y"] = vertices_f32[:, 1]
    verts_packed["z"] = vertices_f32[:, 2]

    # Pack mesh header (single mesh, all vertices)
    mesh_hdr = np.zeros(1, dtype=MESH_HEADER.dtype())
    mesh_hdr["vertex_offset"] = 0
    mesh_hdr["vertex_count"] = n
    mesh_hdr["face_offset"] = 0
    mesh_hdr["face_count"] = 0

    # Pack plane as ClipPlane struct (32B)
    plane_packed = np.zeros(1, dtype=CLIP_PLANE.dtype())
    plane_packed["point_x"] = plane_point_f32[0]
    plane_packed["point_y"] = plane_point_f32[1]
    plane_packed["point_z"] = plane_point_f32[2]
    plane_packed["normal_x"] = plane_normal_f32[0]
    plane_packed["normal_y"] = plane_normal_f32[1]
    plane_packed["normal_z"] = plane_normal_f32[2]

    verts_buf = _buffer_from_numpy(worker, verts_packed)
    mesh_buf = _buffer_from_numpy(worker, mesh_hdr)
    plane_buf = _buffer_from_numpy(worker, plane_packed)
    norm_buf = _rw_buffer(worker, NORM_PARAMS.stride)
    norm_params = _uniform_buffer(worker, grid_bits)

    # Pass 1: normalization reduction
    _dispatch_pass(
        worker,
        "classify_norm",
        [verts_buf, mesh_buf, plane_buf, norm_buf, norm_params],
        (1, 1, 1),
    )

    # Pass 2: per-vertex classification
    signs_buf = _output_buffer(worker, n * 4)
    dots_buf = _output_buffer(worker, n * 4)
    classify_params = _uniform_buffer(worker, n, 0)  # vertex_count, pair_index

    num_workgroups = math.ceil(n / 256)
    _dispatch_pass(
        worker,
        "classify",
        [verts_buf, plane_buf, norm_buf, signs_buf, dots_buf, classify_params],
        (num_workgroups, 1, 1),
    )

    # Read results
    operation = worker.begin_operation()
    signs_raw = worker.read_buffer(signs_buf, n * 4)
    dots_raw = worker.read_buffer(dots_buf, n * 4)
    worker.check_operation(operation)

    signs = np.frombuffer(signs_raw, dtype=np.int32).copy()
    world_dots = np.frombuffer(dots_raw, dtype=np.float32).copy()

    return signs, world_dots
