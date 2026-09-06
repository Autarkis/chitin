"""GPU clip pipeline - unified dispatch chaining all stages.

Public entry point: ``dispatch_clip_pipeline(worker, vertices, faces, planes)``.

GPU-resident chain (zero intermediate host readback):
    classify -> count -> scan -> pack -> emit -> weld_stamp -> weld_sort ->
    weld_dedup -> weld_scan -> weld_finalize -> weld_rewrite

After the chain, a single readback retrieves the arena, welded positions,
scan totals, and weld map.  CPU post-processing (compact, loop extraction,
cap triangulation) runs on the readback data.
"""

from __future__ import annotations

import math
import struct

import numpy as np

from chitin.gpu.caps import CapResult, triangulate_caps
from chitin.gpu.dispatch_multi import prefix_sum_gpu
from chitin.gpu.layouts import CLIP_PLANE, MESH_HEADER, NORM_PARAMS, POINT, TRIANGLE
from chitin.gpu.loops import LoopExtractionResult
from chitin.gpu.pipeline import (
    ClipDiagnostics,
    ClipMetrics,
    ClipPipelineResult,
    ClipStatus,
    validate_clip_inputs,
)
from chitin.gpu.shaders import load_shader
from chitin.gpu.weld import WeldResult, _dedupe_boundary_by_parity
from chitin.gpu.worker import GPUWorker
from chitin.surface_boundary import surface_loops

_STORAGE = 0x80
_COPY_SRC = 0x04
_COPY_DST = 0x08
_UNIFORM = 0x40


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _rw_buffer(worker: GPUWorker, size_bytes: int) -> object:
    return worker.create_buffer(max(size_bytes, 4), _STORAGE | _COPY_SRC | _COPY_DST)


def _uniform_buffer(worker: GPUWorker, data: bytes) -> object:
    padded = data + b"\x00" * (16 - len(data) % 16) if len(data) % 16 else data
    buf = worker.create_buffer(len(padded), _UNIFORM | _COPY_DST)
    worker.write_buffer(buf, padded)
    return buf


def _buffer_from_numpy(worker: GPUWorker, array: np.ndarray) -> object:
    data = array.tobytes()
    buf = worker.create_buffer(max(len(data), 4), _STORAGE | _COPY_DST)
    if data:
        worker.write_buffer(buf, data)
    return buf


def _dispatch_pass(
    worker: GPUWorker,
    shader_name: str,
    buffers: list[object],
    workgroups: tuple[int, int, int],
) -> None:
    pipeline = worker.create_compute_pipeline(load_shader(shader_name))
    bind_group = worker.create_bind_group(
        pipeline,
        0,
        [{"binding": i, "resource": {"buffer": buf}} for i, buf in enumerate(buffers)],
    )
    worker.dispatch_compute(pipeline, [bind_group], workgroups)


def _compute_arena_layout(v_count: int, f_count: int) -> dict[str, int]:
    max_pos = 2 * f_count
    max_neg = 2 * f_count
    max_ixn = 2 * f_count
    max_ixn_padded = _next_power_of_2(max_ixn) if max_ixn > 0 else 0
    max_bnd = f_count

    pos_face_base = 0
    neg_face_base = pos_face_base + max_pos * 4
    ixn_base = neg_face_base + max_neg * 4
    bnd_base = ixn_base + max_ixn_padded * 8
    pos_ancestry_base = bnd_base + max_bnd * 2
    neg_ancestry_base = pos_ancestry_base + max_pos
    arena_total_u32 = neg_ancestry_base + max_neg

    return {
        "pos_face_base": pos_face_base,
        "neg_face_base": neg_face_base,
        "ixn_base": ixn_base,
        "bnd_base": bnd_base,
        "pos_ancestry_base": pos_ancestry_base,
        "neg_ancestry_base": neg_ancestry_base,
        "arena_total_u32": arena_total_u32,
        "max_pos": max_pos,
        "max_neg": max_neg,
        "max_ixn": max_ixn,
        "max_ixn_padded": max_ixn_padded,
        "max_bnd": max_bnd,
    }


def _compact_side(
    source_vertices: np.ndarray,
    faces: np.ndarray,
    welded_positions: np.ndarray,
    v_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float32), faces

    source_mask = faces < v_count
    referenced_source = np.unique(faces[source_mask])

    compact_map = np.full(v_count, -1, dtype=np.int32)
    compact_map[referenced_source] = np.arange(len(referenced_source), dtype=np.int32)

    compact_verts = source_vertices[referenced_source]

    ixn_mask = ~source_mask
    if np.any(ixn_mask):
        ixn_indices = np.unique(faces[ixn_mask]) - v_count
        welded_verts = welded_positions[ixn_indices]
        all_verts = np.concatenate([compact_verts, welded_verts], axis=0)
        ixn_remap = np.full(len(welded_positions), -1, dtype=np.int32)
        ixn_remap[ixn_indices] = np.arange(
            len(referenced_source),
            len(referenced_source) + len(ixn_indices),
            dtype=np.int32,
        )
    else:
        all_verts = compact_verts
        ixn_remap = np.array([], dtype=np.int32)

    new_faces = np.empty_like(faces)
    for col in range(3):
        col_vals = faces[:, col]
        is_src = col_vals < v_count
        new_faces[is_src, col] = compact_map[col_vals[is_src]]
        is_ixn = ~is_src
        if np.any(is_ixn):
            new_faces[is_ixn, col] = ixn_remap[col_vals[is_ixn] - v_count]

    return all_verts, new_faces


def dispatch_clip_pipeline_batch(
    worker: GPUWorker,
    vertices: np.ndarray,
    faces: np.ndarray,
    planes: list[tuple[np.ndarray, np.ndarray]],
    mesh_headers: list[tuple[int, int, int, int]],
) -> list[ClipPipelineResult]:
    """Dispatch the clip pipeline for every (mesh, plane) pair in a batch.

    ``vertices``/``faces`` are the concatenation of all meshes; each face's
    indices are global (index directly into the concatenated ``vertices``
    array). ``mesh_headers`` gives the ``(vertex_offset, vertex_count,
    face_offset, face_count)`` slice for each mesh. Results are ordered with
    the mesh as the outer loop and the plane as the inner loop.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)

    for v_off, v_count, f_off, f_count in mesh_headers:
        if v_off < 0 or v_count < 0 or f_off < 0 or f_count < 0:
            raise ValueError(
                f"mesh header has negative offset/count: {(v_off, v_count, f_off, f_count)}"
            )
        if v_off + v_count > len(vertices):
            raise ValueError(
                f"mesh header vertex range [{v_off}, {v_off + v_count}) exceeds vertices length {len(vertices)}"
            )
        if f_off + f_count > len(faces):
            raise ValueError(
                f"mesh header face range [{f_off}, {f_off + f_count}) exceeds faces length {len(faces)}"
            )

    normalized_planes = []
    for plane_point, plane_normal in planes:
        pp = np.asarray(plane_point, dtype=np.float32).reshape(-1)
        pn = np.asarray(plane_normal, dtype=np.float32).reshape(-1)
        if pp.shape != (3,) or pn.shape != (3,):
            raise ValueError(
                f"plane point/normal must each have 3 elements, got shapes {pp.shape}, {pn.shape}"
            )
        normalized_planes.append((pp, pn))

    results: list[ClipPipelineResult] = []
    for v_off, v_count, f_off, f_count in mesh_headers:
        mesh_verts = vertices[v_off : v_off + v_count]
        mesh_faces_global = faces[f_off : f_off + f_count]
        mesh_faces_local = mesh_faces_global - np.uint32(v_off)
        for plane_point, plane_normal in normalized_planes:
            result = _dispatch_clip_pipeline_impl(
                worker, mesh_verts, mesh_faces_local, plane_point, plane_normal
            )
            results.append(result)
    return results


def _dispatch_clip_pipeline_impl(
    worker: GPUWorker,
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> ClipPipelineResult:
    bad = validate_clip_inputs(vertices, faces, plane_point, plane_normal)
    if bad is not None:
        return ClipPipelineResult.failure(bad)

    vertices_f32 = np.asarray(vertices, dtype=np.float32)
    faces_u32 = np.asarray(faces, dtype=np.uint32)
    v_count = len(vertices_f32)
    f_count = len(faces_u32)

    if f_count == 0:
        return ClipPipelineResult.failure(ClipStatus.ALL_COPLANAR)

    layout = _compute_arena_layout(v_count, f_count)
    arena_bytes = layout["arena_total_u32"] * 4
    if arena_bytes > worker.limits.max_buffer_size:
        return ClipPipelineResult.failure(ClipStatus.CAPACITY_EXCEEDED)

    plane_point_f32 = np.asarray(plane_point, dtype=np.float32).ravel()
    plane_normal_f32 = np.asarray(plane_normal, dtype=np.float32).ravel()

    verts_packed = np.zeros(v_count, dtype=POINT.dtype())
    verts_packed["x"] = vertices_f32[:, 0]
    verts_packed["y"] = vertices_f32[:, 1]
    verts_packed["z"] = vertices_f32[:, 2]

    faces_packed = np.zeros(f_count, dtype=TRIANGLE.dtype())
    faces_packed["i0"] = faces_u32[:, 0]
    faces_packed["i1"] = faces_u32[:, 1]
    faces_packed["i2"] = faces_u32[:, 2]

    verts_buf = _buffer_from_numpy(worker, verts_packed)
    faces_buf = _buffer_from_numpy(worker, faces_packed)

    mesh_hdr = np.zeros(1, dtype=MESH_HEADER.dtype())
    mesh_hdr["vertex_offset"] = 0
    mesh_hdr["vertex_count"] = v_count
    mesh_hdr["face_offset"] = 0
    mesh_hdr["face_count"] = 0
    mesh_buf = _buffer_from_numpy(worker, mesh_hdr)

    plane_packed = np.zeros(1, dtype=CLIP_PLANE.dtype())
    plane_packed["point_x"] = plane_point_f32[0]
    plane_packed["point_y"] = plane_point_f32[1]
    plane_packed["point_z"] = plane_point_f32[2]
    plane_packed["normal_x"] = plane_normal_f32[0]
    plane_packed["normal_y"] = plane_normal_f32[1]
    plane_packed["normal_z"] = plane_normal_f32[2]
    plane_buf = _buffer_from_numpy(worker, plane_packed)

    norm_buf = _rw_buffer(worker, NORM_PARAMS.stride)
    norm_params = _uniform_buffer(worker, struct.pack("<I", 20))

    _dispatch_pass(
        worker,
        "classify_norm",
        [verts_buf, mesh_buf, plane_buf, norm_buf, norm_params],
        (1, 1, 1),
    )

    signs_buf = _rw_buffer(worker, v_count * 4)
    dots_buf = _rw_buffer(worker, v_count * 4)
    classify_counters_buf = _rw_buffer(worker, 8)
    classify_params = _uniform_buffer(worker, struct.pack("<IIII", v_count, 0, 0, 0))

    classify_wg = math.ceil(v_count / 256)
    _dispatch_pass(
        worker,
        "classify",
        [
            verts_buf,
            plane_buf,
            norm_buf,
            signs_buf,
            dots_buf,
            classify_params,
            classify_counters_buf,
        ],
        (classify_wg, 1, 1),
    )

    pos_count_buf = _rw_buffer(worker, f_count * 4)
    neg_count_buf = _rw_buffer(worker, f_count * 4)
    ixn_count_buf = _rw_buffer(worker, f_count * 4)
    bnd_count_buf = _rw_buffer(worker, f_count * 4)
    count_params = _uniform_buffer(worker, struct.pack("<IIII", f_count, 0, 0, 0))

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

    pos_off_buf, pos_total_buf = prefix_sum_gpu(worker, pos_count_buf, f_count)
    neg_off_buf, neg_total_buf = prefix_sum_gpu(worker, neg_count_buf, f_count)
    ixn_off_buf, ixn_total_buf = prefix_sum_gpu(worker, ixn_count_buf, f_count)
    bnd_off_buf, bnd_total_buf = prefix_sum_gpu(worker, bnd_count_buf, f_count)

    packed_offsets_buf = _rw_buffer(worker, 4 * f_count * 4)
    pack_params = _uniform_buffer(worker, struct.pack("<IIII", f_count, 0, 0, 0))
    _dispatch_pass(
        worker,
        "pack_offsets",
        [
            pos_off_buf,
            neg_off_buf,
            ixn_off_buf,
            bnd_off_buf,
            packed_offsets_buf,
            pack_params,
        ],
        (num_wg, 1, 1),
    )

    arena_buf = _rw_buffer(worker, max(layout["arena_total_u32"] * 4, 4))
    emit_params = _uniform_buffer(
        worker,
        struct.pack(
            "<IIIIIIII",
            f_count,
            v_count,
            layout["pos_face_base"],
            layout["neg_face_base"],
            layout["ixn_base"],
            layout["bnd_base"],
            layout["pos_ancestry_base"],
            layout["neg_ancestry_base"],
        ),
    )
    _dispatch_pass(
        worker,
        "clip_emit",
        [
            verts_buf,
            faces_buf,
            signs_buf,
            dots_buf,
            packed_offsets_buf,
            arena_buf,
            emit_params,
        ],
        (num_wg, 1, 1),
    )

    max_ixn_padded = layout["max_ixn_padded"]
    ixn_base = layout["ixn_base"]
    welded_pos_buf = None
    weld_total_buf = None

    if max_ixn_padded > 0:
        stamp_params = _uniform_buffer(
            worker, struct.pack("<IIII", ixn_base, max_ixn_padded, 0, 0)
        )
        stamp_wg = math.ceil(max_ixn_padded / 256)
        _dispatch_pass(
            worker, "weld_stamp", [arena_buf, stamp_params], (stamp_wg, 1, 1)
        )

        n_sort = max_ixn_padded
        sort_wg = max(math.ceil(n_sort // 2 / 256), 1)
        k = 2
        while k <= n_sort:
            j = k >> 1
            while j >= 1:
                sort_params = _uniform_buffer(
                    worker, struct.pack("<IIII", ixn_base, n_sort, k, j)
                )
                _dispatch_pass(
                    worker,
                    "weld_sort_step",
                    [arena_buf, sort_params],
                    (sort_wg, 1, 1),
                )
                j >>= 1
            k <<= 1

        unique_flags_buf = _rw_buffer(worker, max_ixn_padded * 4)
        dedup_params = _uniform_buffer(
            worker, struct.pack("<IIII", ixn_base, max_ixn_padded, 0, 0)
        )
        _dispatch_pass(
            worker,
            "weld_dedup",
            [arena_buf, unique_flags_buf, dedup_params],
            (stamp_wg, 1, 1),
        )

        weld_scan_buf, weld_total_buf = prefix_sum_gpu(
            worker, unique_flags_buf, max_ixn_padded
        )

        weld_map_buf = _rw_buffer(worker, max_ixn_padded * 4)
        welded_pos_buf = _rw_buffer(worker, max_ixn_padded * 16)
        finalize_params = _uniform_buffer(
            worker, struct.pack("<IIII", ixn_base, max_ixn_padded, 0, 0)
        )
        _dispatch_pass(
            worker,
            "weld_finalize",
            [
                arena_buf,
                weld_scan_buf,
                unique_flags_buf,
                weld_map_buf,
                welded_pos_buf,
                finalize_params,
            ],
            (stamp_wg, 1, 1),
        )

        max_pos = layout["max_pos"]
        max_neg = layout["max_neg"]
        max_bnd = layout["max_bnd"]
        rewrite_thread_count = max_pos + max_neg + max_bnd
        rewrite_wg = max(math.ceil(rewrite_thread_count / 256), 1)
        rewrite_params = _uniform_buffer(
            worker,
            struct.pack(
                "<IIIIIIII",
                v_count,
                layout["pos_face_base"],
                max_pos,
                layout["neg_face_base"],
                max_neg,
                layout["bnd_base"],
                max_bnd,
                0,
            ),
        )
        _dispatch_pass(
            worker,
            "weld_rewrite",
            [arena_buf, weld_map_buf, rewrite_params],
            (rewrite_wg, 1, 1),
        )

    # ===== SINGLE READBACK =====
    operation = worker.begin_operation()
    arena_raw = worker.read_buffer(arena_buf, layout["arena_total_u32"] * 4)
    pos_total_raw = worker.read_buffer(pos_total_buf, 4)
    neg_total_raw = worker.read_buffer(neg_total_buf, 4)
    ixn_total_raw = worker.read_buffer(ixn_total_buf, 4)
    bnd_total_raw = worker.read_buffer(bnd_total_buf, 4)

    welded_pos_raw = None
    weld_total_raw = None
    if welded_pos_buf is not None:
        welded_pos_raw = worker.read_buffer(welded_pos_buf, max_ixn_padded * 16)
        weld_total_raw = worker.read_buffer(weld_total_buf, 4)
    signs_raw = worker.read_buffer(signs_buf, v_count * 4)
    counters_raw = worker.read_buffer(classify_counters_buf, 8)
    worker.check_operation(operation)

    total_pos = int(np.frombuffer(pos_total_raw, dtype=np.int32)[0])
    total_neg = int(np.frombuffer(neg_total_raw, dtype=np.int32)[0])
    total_ixn = int(np.frombuffer(ixn_total_raw, dtype=np.int32)[0])
    total_bnd = int(np.frombuffer(bnd_total_raw, dtype=np.int32)[0])

    if total_pos == 0 and total_neg == 0 and total_ixn == 0:
        signs = np.frombuffer(signs_raw, dtype=np.int32).copy()
        return ClipPipelineResult(
            status=ClipStatus.ALL_COPLANAR,
            positive_vertices=np.zeros((0, 3), dtype=np.float32),
            positive_faces=np.zeros((0, 3), dtype=np.uint32),
            negative_vertices=np.zeros((0, 3), dtype=np.float32),
            negative_faces=np.zeros((0, 3), dtype=np.uint32),
            positive_ancestry=np.array([], dtype=np.uint32),
            negative_ancestry=np.array([], dtype=np.uint32),
            diagnostics=ClipDiagnostics(signs=signs),
        )
    if total_neg == 0 and total_ixn == 0:
        signs = np.frombuffer(signs_raw, dtype=np.int32).copy()
        return ClipPipelineResult(
            status=ClipStatus.ALL_POSITIVE,
            positive_vertices=vertices_f32.copy(),
            positive_faces=faces_u32.copy(),
            negative_vertices=np.zeros((0, 3), dtype=np.float32),
            negative_faces=np.zeros((0, 3), dtype=np.uint32),
            positive_ancestry=np.arange(f_count, dtype=np.uint32),
            negative_ancestry=np.array([], dtype=np.uint32),
            metrics=ClipMetrics(
                input_face_count=f_count,
                positive_face_count=f_count,
            ),
            diagnostics=ClipDiagnostics(signs=signs),
        )
    if total_pos == 0 and total_ixn == 0:
        signs = np.frombuffer(signs_raw, dtype=np.int32).copy()
        return ClipPipelineResult(
            status=ClipStatus.ALL_NEGATIVE,
            positive_vertices=np.zeros((0, 3), dtype=np.float32),
            positive_faces=np.zeros((0, 3), dtype=np.uint32),
            negative_vertices=vertices_f32.copy(),
            negative_faces=faces_u32.copy(),
            positive_ancestry=np.array([], dtype=np.uint32),
            negative_ancestry=np.arange(f_count, dtype=np.uint32),
            metrics=ClipMetrics(
                input_face_count=f_count,
                negative_face_count=f_count,
            ),
            diagnostics=ClipDiagnostics(signs=signs),
        )

    arena = np.frombuffer(arena_raw, dtype=np.uint32).copy()
    L = layout

    if total_pos > 0:
        pf = arena[L["pos_face_base"] : L["pos_face_base"] + total_pos * 4]
        pos_faces = pf.reshape(total_pos, 4)[:, :3].copy()
        pos_ancestry = arena[
            L["pos_ancestry_base"] : L["pos_ancestry_base"] + total_pos
        ].copy()
    else:
        pos_faces = np.zeros((0, 3), dtype=np.uint32)
        pos_ancestry = np.array([], dtype=np.uint32)

    if total_neg > 0:
        nf = arena[L["neg_face_base"] : L["neg_face_base"] + total_neg * 4]
        neg_faces = nf.reshape(total_neg, 4)[:, :3].copy()
        neg_ancestry = arena[
            L["neg_ancestry_base"] : L["neg_ancestry_base"] + total_neg
        ].copy()
    else:
        neg_faces = np.zeros((0, 3), dtype=np.uint32)
        neg_ancestry = np.array([], dtype=np.uint32)

    if total_bnd > 0:
        bf = arena[L["bnd_base"] : L["bnd_base"] + total_bnd * 2]
        boundary_edges = bf.reshape(total_bnd, 2).copy()
    else:
        boundary_edges = np.zeros((0, 2), dtype=np.uint32)

    if weld_total_raw is not None and total_ixn > 0:
        total_welded = int(np.frombuffer(weld_total_raw, dtype=np.int32)[0])
        wp = np.frombuffer(welded_pos_raw, dtype=np.uint32).copy()
        welded_positions = (
            wp[: total_welded * 4]
            .reshape(total_welded, 4)[:, :3]
            .copy()
            .view(np.float32)
        )
        ixn_start = layout["ixn_base"]
        ixn_end = ixn_start + max_ixn_padded * 8
        ixn_records = arena[ixn_start:ixn_end].reshape(max_ixn_padded, 8)
        all_keys = ixn_records[:, [4, 5]].copy()
        _, first_idx = np.unique(
            all_keys[:, 0].astype(np.uint64) << 32 | all_keys[:, 1].astype(np.uint64),
            return_index=True,
        )
        first_idx.sort()
        welded_edge_keys = all_keys[first_idx[:total_welded]].copy()
    else:
        total_welded = 0
        welded_positions = np.zeros((0, 3), dtype=np.float32)
        welded_edge_keys = np.zeros((0, 2), dtype=np.uint32)

    boundary_edges_deduped = _dedupe_boundary_by_parity(boundary_edges)

    weld_result = WeldResult(
        positive_faces=pos_faces,
        negative_faces=neg_faces,
        welded_positions=welded_positions,
        welded_edge_keys=welded_edge_keys,
        boundary_edges=boundary_edges_deduped,
        positive_ancestry=pos_ancestry,
        negative_ancestry=neg_ancestry,
        intersection_count_raw=total_ixn,
        intersection_count_welded=total_welded,
    )

    signs = np.frombuffer(signs_raw, dtype=np.int32).copy()

    def side_loops(side_faces):
        try:
            loops = surface_loops(side_faces)
            for loop in loops:
                source = loop[loop < v_count]
                if np.any(signs[source] != 0):
                    raise ValueError("input has an open boundary away from cut plane")
            return LoopExtractionResult(loops, True)
        except ValueError:
            return LoopExtractionResult([], False)

    loop_result = side_loops(pos_faces)
    negative_loop_result = side_loops(neg_faces)
    if not loop_result.valid or not negative_loop_result.valid:
        signs = np.frombuffer(signs_raw, dtype=np.int32).copy()
        return ClipPipelineResult(
            status=ClipStatus.TOPOLOGY_ERROR,
            positive_vertices=np.zeros((0, 3), dtype=np.float32),
            positive_faces=np.zeros((0, 3), dtype=np.uint32),
            negative_vertices=np.zeros((0, 3), dtype=np.float32),
            negative_faces=np.zeros((0, 3), dtype=np.uint32),
            positive_ancestry=np.array([], dtype=np.uint32),
            negative_ancestry=np.array([], dtype=np.uint32),
            diagnostics=ClipDiagnostics(
                signs=signs,
                weld_result=weld_result,
                loop_result=loop_result,
                boundary_edges_raw=boundary_edges,
            ),
        )

    positive_cap = triangulate_caps(
        loop_result.loops, weld_result, vertices_f32, plane_normal_f32
    )
    negative_cap = triangulate_caps(
        negative_loop_result.loops, weld_result, vertices_f32, plane_normal_f32
    )
    cap_result = CapResult(
        positive_cap.positive_cap_faces,
        negative_cap.negative_cap_faces,
        len(loop_result.loops) + len(negative_loop_result.loops),
        len(positive_cap.positive_cap_faces) + len(negative_cap.negative_cap_faces),
        (positive_cap.valid or not loop_result.loops)
        and (negative_cap.valid or not negative_loop_result.loops),
    )

    if not cap_result.valid:
        return ClipPipelineResult.failure(ClipStatus.TOPOLOGY_ERROR)

    if cap_result.valid and cap_result.total_cap_faces > 0:
        if len(cap_result.positive_cap_faces) > 0:
            pos_faces = np.concatenate(
                [pos_faces, cap_result.positive_cap_faces], axis=0
            )
            pos_ancestry = np.concatenate(
                [
                    pos_ancestry,
                    np.full(
                        len(cap_result.positive_cap_faces), 0xFFFFFFFF, dtype=np.uint32
                    ),
                ]
            )
        if len(cap_result.negative_cap_faces) > 0:
            neg_faces = np.concatenate(
                [neg_faces, cap_result.negative_cap_faces], axis=0
            )
            neg_ancestry = np.concatenate(
                [
                    neg_ancestry,
                    np.full(
                        len(cap_result.negative_cap_faces), 0xFFFFFFFF, dtype=np.uint32
                    ),
                ]
            )

    # Compact AFTER cap concatenation so cap faces get remapped too.
    pos_verts, pos_faces = _compact_side(
        vertices_f32, pos_faces, welded_positions, v_count
    )
    neg_verts, neg_faces = _compact_side(
        vertices_f32, neg_faces, welded_positions, v_count
    )

    signs = np.frombuffer(signs_raw, dtype=np.int32).copy()
    classify_counts = np.frombuffer(counters_raw, dtype=np.uint32).copy()
    fallback_count = int(classify_counts[0])

    on_plane = int(np.sum(signs == 0))
    coplanar = int(
        np.sum(
            (signs[faces_u32[:, 0]] == 0)
            & (signs[faces_u32[:, 1]] == 0)
            & (signs[faces_u32[:, 2]] == 0)
        )
    )

    metrics = ClipMetrics(
        classify_fast_count=v_count - fallback_count - on_plane,
        classify_fallback_count=fallback_count,
        classify_on_plane_count=on_plane,
        input_face_count=f_count,
        positive_face_count=len(pos_faces),
        negative_face_count=len(neg_faces),
        coplanar_face_count=coplanar,
        intersection_count_raw=total_ixn,
        intersection_count_welded=total_welded,
        boundary_loop_count=len(loop_result.loops),
        cap_face_count=cap_result.total_cap_faces if cap_result.valid else 0,
    )

    if total_welded > 0:
        real_ixn = welded_edge_keys[:, 0] != welded_edge_keys[:, 1]
        n_real = int(real_ixn.sum())
        ixn_rec = np.zeros((n_real, 5), dtype=np.float32)
        ixn_rec[:, :3] = welded_positions[real_ixn]
        ixn_rec[:, 3] = welded_edge_keys[real_ixn, 0].astype(np.float32)
        ixn_rec[:, 4] = welded_edge_keys[real_ixn, 1].astype(np.float32)
    else:
        ixn_rec = np.zeros((0, 5), dtype=np.float32)

    diag = ClipDiagnostics(
        signs=signs,
        weld_result=weld_result,
        loop_result=loop_result,
        negative_loop_result=negative_loop_result,
        cap_result=cap_result,
        boundary_edges_raw=boundary_edges,
    )

    return ClipPipelineResult(
        status=ClipStatus.OK,
        positive_vertices=pos_verts,
        positive_faces=pos_faces,
        negative_vertices=neg_verts,
        negative_faces=neg_faces,
        positive_ancestry=pos_ancestry,
        negative_ancestry=neg_ancestry,
        intersection_records=ixn_rec,
        metrics=metrics,
        diagnostics=diag,
    )


def dispatch_clip_pipeline(
    worker: GPUWorker,
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> ClipPipelineResult:
    """Dispatch the clip pipeline for a single mesh against a single plane."""
    results = dispatch_clip_pipeline_batch(
        worker,
        vertices,
        faces,
        [(plane_point, plane_normal)],
        [(0, len(vertices), 0, len(faces))],
    )
    return results[0]
