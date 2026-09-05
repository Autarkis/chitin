"""GPU clip stage — count/scan/emit pipeline for plane-clipping a mesh.

Three-pass pipeline:
1. clip_count: per-triangle output-geometry counts (positive/negative faces,
   intersections, boundary edges), driven by the classify stage's signs.
2. scan: exclusive prefix sum of each count array, giving per-triangle
   arena write offsets plus the grand totals.
3. clip_emit: per-triangle emission of clipped geometry into a packed
   output arena, using the scanned offsets.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np

from chitin.gpu.dispatch_multi import dispatch_prefix_sum_exclusive_multi
from chitin.gpu.layouts import INTERSECTION_RECORD, POINT, TRIANGLE
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


def _uniform_buffer_raw(worker: GPUWorker, data: bytes) -> object:
    """Build a uniform buffer holding an arbitrary struct, 16-byte aligned."""
    size = max(len(data), 16)
    size = (size + 15) & ~15
    padded = data + b"\x00" * (size - len(data))
    buffer = worker.create_buffer(size, _UNIFORM | _COPY_DST)
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


@dataclass
class ClipEmitResult:
    positive_faces: np.ndarray  # (N, 3) uint32 — global vertex indices
    negative_faces: np.ndarray  # (N, 3) uint32 — global vertex indices
    intersection_positions: np.ndarray  # (N, 3) float32 — xyz positions
    intersection_edges: np.ndarray  # (N, 2) uint32 — (v_lo, v_hi) EdgeKeys
    boundary_edges: np.ndarray  # (N, 2) uint32 — vertex index pairs
    positive_ancestry: np.ndarray  # (N,) uint32 — source face index per pos face
    negative_ancestry: np.ndarray  # (N,) uint32 — source face index per neg face
    total_positive_faces: int
    total_negative_faces: int
    total_intersections: int
    total_boundary_edges: int


def _empty_result() -> ClipEmitResult:
    return ClipEmitResult(
        positive_faces=np.zeros((0, 3), dtype=np.uint32),
        negative_faces=np.zeros((0, 3), dtype=np.uint32),
        intersection_positions=np.zeros((0, 3), dtype=np.float32),
        intersection_edges=np.zeros((0, 2), dtype=np.uint32),
        boundary_edges=np.zeros((0, 2), dtype=np.uint32),
        positive_ancestry=np.array([], dtype=np.uint32),
        negative_ancestry=np.array([], dtype=np.uint32),
        total_positive_faces=0,
        total_negative_faces=0,
        total_intersections=0,
        total_boundary_edges=0,
    )


def dispatch_clip_emit_reference(
    worker: GPUWorker,
    vertices: np.ndarray,
    faces: np.ndarray,
    signs: np.ndarray,
    world_dots: np.ndarray,
) -> ClipEmitResult:
    """Clip a mesh against a plane on GPU, given classify-stage signs/dots.

    ``vertices`` is (V, 3) float32, ``faces`` is (F, 3) uint32, ``signs`` is
    (V,) int32 in {-1, 0, +1}, and ``world_dots`` is (V,) float32 world-frame
    signed distances (both from ``dispatch_classify``). Returns a
    ``ClipEmitResult`` with the positive-side mesh, negative-side mesh,
    intersection points/edges, boundary edges, and per-face ancestry.
    """
    vertices_f32 = np.asarray(vertices, dtype=np.float32)
    if vertices_f32.ndim != 2 or vertices_f32.shape[1] != 3:
        raise ValueError("vertices must be (V, 3)")
    faces_u32 = np.asarray(faces, dtype=np.uint32)
    if faces_u32.ndim != 2 or faces_u32.shape[1] != 3:
        raise ValueError("faces must be (F, 3)")
    signs_i32 = np.asarray(signs, dtype=np.int32).ravel()
    if signs_i32.shape != (len(vertices_f32),):
        raise ValueError("signs must be (V,)")
    world_dots_f32 = np.asarray(world_dots, dtype=np.float32).ravel()
    if world_dots_f32.shape != (len(vertices_f32),):
        raise ValueError("world_dots must be (V,)")

    v_count = len(vertices_f32)
    f_count = len(faces_u32)
    if f_count == 0:
        return _empty_result()

    # Pack vertices as Point structs (16B each).
    verts_packed = np.zeros(v_count, dtype=POINT.dtype())
    verts_packed["x"] = vertices_f32[:, 0]
    verts_packed["y"] = vertices_f32[:, 1]
    verts_packed["z"] = vertices_f32[:, 2]

    # Pack faces as Triangle structs (16B each).
    faces_packed = np.zeros(f_count, dtype=TRIANGLE.dtype())
    faces_packed["i0"] = faces_u32[:, 0]
    faces_packed["i1"] = faces_u32[:, 1]
    faces_packed["i2"] = faces_u32[:, 2]

    verts_buf = _buffer_from_numpy(worker, verts_packed)
    faces_buf = _buffer_from_numpy(worker, faces_packed)
    signs_buf = _buffer_from_numpy(worker, signs_i32)
    dots_buf = _buffer_from_numpy(worker, world_dots_f32)

    # --- Pass 1: count ---
    pos_count_buf = _rw_buffer(worker, f_count * 4)
    neg_count_buf = _rw_buffer(worker, f_count * 4)
    ixn_count_buf = _rw_buffer(worker, f_count * 4)
    bnd_count_buf = _rw_buffer(worker, f_count * 4)
    count_params = _uniform_buffer_raw(worker, struct.pack("<IIII", f_count, 0, 0, 0))

    num_wg = math.ceil(f_count / 256)
    _dispatch_pass(
        worker,
        "clip_count",
        [
            faces_buf,
            signs_buf,
            pos_count_buf,
            neg_count_buf,
            ixn_count_buf,
            bnd_count_buf,
            count_params,
        ],
        (num_wg, 1, 1),
    )

    operation = worker.begin_operation()
    pos_raw = worker.read_buffer(pos_count_buf, f_count * 4)
    neg_raw = worker.read_buffer(neg_count_buf, f_count * 4)
    ixn_raw = worker.read_buffer(ixn_count_buf, f_count * 4)
    bnd_raw = worker.read_buffer(bnd_count_buf, f_count * 4)
    worker.check_operation(operation)

    pos_counts = np.frombuffer(pos_raw, dtype=np.int32).copy()
    neg_counts = np.frombuffer(neg_raw, dtype=np.int32).copy()
    ixn_counts = np.frombuffer(ixn_raw, dtype=np.int32).copy()
    bnd_counts = np.frombuffer(bnd_raw, dtype=np.int32).copy()

    # --- Pass 2: scan ---
    pos_offsets, total_pos = dispatch_prefix_sum_exclusive_multi(worker, pos_counts)
    neg_offsets, total_neg = dispatch_prefix_sum_exclusive_multi(worker, neg_counts)
    ixn_offsets, total_ixn = dispatch_prefix_sum_exclusive_multi(worker, ixn_counts)
    bnd_offsets, total_bnd = dispatch_prefix_sum_exclusive_multi(worker, bnd_counts)

    if total_pos == 0 and total_neg == 0:
        return _empty_result()

    packed_offsets = np.concatenate(
        [
            pos_offsets.astype(np.int32),
            neg_offsets.astype(np.int32),
            ixn_offsets.astype(np.int32),
            bnd_offsets.astype(np.int32),
        ]
    )
    offsets_buf = _buffer_from_numpy(worker, packed_offsets)

    # --- Output arena layout (all offsets in u32 units) ---
    pos_face_base = 0
    neg_face_base = pos_face_base + total_pos * 4  # 4 u32 per Triangle
    ixn_base = neg_face_base + total_neg * 4
    bnd_base = ixn_base + total_ixn * 8  # 8 u32 per IntersectionRecord
    pos_ancestry_base = bnd_base + total_bnd * 2  # 2 u32 per EdgeKey
    neg_ancestry_base = pos_ancestry_base + total_pos  # 1 u32 per ancestry entry
    arena_total_u32 = neg_ancestry_base + total_neg

    arena_buf = _rw_buffer(worker, max(arena_total_u32 * 4, 4))

    emit_params = _uniform_buffer_raw(
        worker,
        struct.pack(
            "<IIIIIIII",
            f_count,
            v_count,
            pos_face_base,
            neg_face_base,
            ixn_base,
            bnd_base,
            pos_ancestry_base,
            neg_ancestry_base,
        ),
    )

    # --- Pass 3: emit ---
    _dispatch_pass(
        worker,
        "clip_emit",
        [
            verts_buf,
            faces_buf,
            signs_buf,
            dots_buf,
            offsets_buf,
            arena_buf,
            emit_params,
        ],
        (num_wg, 1, 1),
    )

    operation = worker.begin_operation()
    arena_raw = worker.read_buffer(arena_buf, arena_total_u32 * 4)
    worker.check_operation(operation)

    arena = np.frombuffer(arena_raw, dtype=np.uint32).copy()

    if total_pos > 0:
        pos_flat = arena[pos_face_base : pos_face_base + total_pos * 4]
        pos_faces = pos_flat.reshape(total_pos, 4)[:, :3].copy()
    else:
        pos_faces = np.zeros((0, 3), dtype=np.uint32)

    if total_neg > 0:
        neg_flat = arena[neg_face_base : neg_face_base + total_neg * 4]
        neg_faces = neg_flat.reshape(total_neg, 4)[:, :3].copy()
    else:
        neg_faces = np.zeros((0, 3), dtype=np.uint32)

    if total_ixn > 0:
        ixn_flat = arena[
            ixn_base : ixn_base + total_ixn * INTERSECTION_RECORD.stride // 4
        ]
        ixn_flat = ixn_flat.reshape(total_ixn, INTERSECTION_RECORD.stride // 4)
        ixn_pos = ixn_flat[:, :3].copy().view(np.float32)
        ixn_edges = ixn_flat[:, 4:6].copy()
    else:
        ixn_pos = np.zeros((0, 3), dtype=np.float32)
        ixn_edges = np.zeros((0, 2), dtype=np.uint32)

    if total_bnd > 0:
        bnd_flat = arena[bnd_base : bnd_base + total_bnd * 2]
        boundary_edges = bnd_flat.reshape(total_bnd, 2).copy()
    else:
        boundary_edges = np.zeros((0, 2), dtype=np.uint32)

    if total_pos > 0:
        pos_ancestry = arena[pos_ancestry_base : pos_ancestry_base + total_pos].copy()
    else:
        pos_ancestry = np.array([], dtype=np.uint32)

    if total_neg > 0:
        neg_ancestry = arena[neg_ancestry_base : neg_ancestry_base + total_neg].copy()
    else:
        neg_ancestry = np.array([], dtype=np.uint32)

    return ClipEmitResult(
        positive_faces=pos_faces,
        negative_faces=neg_faces,
        intersection_positions=ixn_pos,
        intersection_edges=ixn_edges,
        boundary_edges=boundary_edges,
        positive_ancestry=pos_ancestry,
        negative_ancestry=neg_ancestry,
        total_positive_faces=total_pos,
        total_negative_faces=total_neg,
        total_intersections=total_ixn,
        total_boundary_edges=total_bnd,
    )
