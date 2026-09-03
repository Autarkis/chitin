"""High-level dispatch helpers: numpy array in, numpy array out.

Each function compiles the shader (cached by GPUWorker), creates buffers,
dispatches, reads back. Single-workgroup only (max 256 elements).
"""

from __future__ import annotations

import struct

import numpy as np

from chitin.gpu.shaders import load_shader
from chitin.gpu.worker import GPUWorker

MAX_WORKGROUP_ELEMENTS = 256

# wgpu buffer usage flags (standard WebGPU values)
_STORAGE = 0x80  # STORAGE
_COPY_SRC = 0x04  # COPY_SRC
_COPY_DST = 0x08  # COPY_DST
_UNIFORM = 0x40  # UNIFORM


def _uniform_buffer(worker: GPUWorker, count: int) -> object:
    """Create a uniform buffer with params vec4<u32> where x=count."""
    data = struct.pack("<IIII", count, 0, 0, 0)
    buf = worker.create_buffer(16, _UNIFORM | _COPY_DST)
    worker._device.queue.write_buffer(buf, 0, data)
    return buf


def _storage_buffer_from_numpy(worker: GPUWorker, arr: np.ndarray) -> object:
    """Create a storage buffer initialized with numpy array data."""
    data = arr.tobytes()
    size = max(len(data), 4)  # WebGPU requires non-zero buffer size
    buf = worker.create_buffer(size, _STORAGE | _COPY_DST)
    worker._device.queue.write_buffer(buf, 0, data)
    return buf


def _output_buffer(worker: GPUWorker, size: int) -> object:
    """Create a storage buffer for output (readable)."""
    size = max(size, 4)
    return worker.create_buffer(size, _STORAGE | _COPY_SRC)


def dispatch_prefix_sum_exclusive(
    worker: GPUWorker, values: np.ndarray
) -> tuple[np.ndarray, int]:
    """Run exclusive prefix sum on GPU. Returns (scan_result, total).

    values: i32 array, max 256 elements.
    Returns: (i32 exclusive scan, i32 total).
    """
    n = len(values)
    if n == 0:
        return np.array([], dtype=np.int32), 0
    if n > MAX_WORKGROUP_ELEMENTS:
        raise ValueError(
            f"prefix_sum: {n} elements exceeds workgroup max {MAX_WORKGROUP_ELEMENTS}"
        )

    values_i32 = values.astype(np.int32)
    wgsl = load_shader("prefix_sum")
    pipeline = worker.create_compute_pipeline(wgsl, "main")

    input_buf = _storage_buffer_from_numpy(worker, values_i32)
    # Output: n elements + 1 for total
    output_buf = _output_buffer(worker, (n + 1) * 4)
    params_buf = _uniform_buffer(worker, n)

    bind_group = worker.create_bind_group(
        pipeline,
        0,
        [
            {"binding": 0, "resource": {"buffer": input_buf}},
            {"binding": 1, "resource": {"buffer": output_buf}},
            {"binding": 2, "resource": {"buffer": params_buf}},
        ],
    )

    worker.dispatch_compute(pipeline, [bind_group], (1, 1, 1))
    raw = worker.read_buffer(output_buf, (n + 1) * 4)
    result = np.frombuffer(raw, dtype=np.int32)
    return result[:n].copy(), int(result[n])


def dispatch_reduce_sum(worker: GPUWorker, values: np.ndarray) -> float:
    """Run tree reduction sum on GPU. Returns f32 sum.

    values: f32 array, max 256 elements.
    """
    n = len(values)
    if n == 0:
        return 0.0
    if n > MAX_WORKGROUP_ELEMENTS:
        raise ValueError(
            f"reduce_sum: {n} elements exceeds workgroup max {MAX_WORKGROUP_ELEMENTS}"
        )

    values_f32 = values.astype(np.float32)
    wgsl = load_shader("reduce_sum")
    pipeline = worker.create_compute_pipeline(wgsl, "main")

    input_buf = _storage_buffer_from_numpy(worker, values_f32)
    output_buf = _output_buffer(worker, 4)
    params_buf = _uniform_buffer(worker, n)

    bind_group = worker.create_bind_group(
        pipeline,
        0,
        [
            {"binding": 0, "resource": {"buffer": input_buf}},
            {"binding": 1, "resource": {"buffer": output_buf}},
            {"binding": 2, "resource": {"buffer": params_buf}},
        ],
    )

    worker.dispatch_compute(pipeline, [bind_group], (1, 1, 1))
    raw = worker.read_buffer(output_buf, 4)
    return float(np.frombuffer(raw, dtype=np.float32)[0])


def dispatch_compact(
    worker: GPUWorker, values: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, int]:
    """Run stream compaction on GPU. Returns (compacted_values, count).

    values: i32 array, mask: i32 array (nonzero = keep). Max 256 elements.
    """
    n = len(values)
    if n == 0:
        return np.array([], dtype=np.int32), 0
    if n > MAX_WORKGROUP_ELEMENTS:
        raise ValueError(
            f"compact: {n} elements exceeds workgroup max {MAX_WORKGROUP_ELEMENTS}"
        )

    values_i32 = values.astype(np.int32)
    mask_i32 = mask.astype(np.int32)
    wgsl = load_shader("compact")
    pipeline = worker.create_compute_pipeline(wgsl, "main")

    val_buf = _storage_buffer_from_numpy(worker, values_i32)
    mask_buf = _storage_buffer_from_numpy(worker, mask_i32)
    out_buf = _output_buffer(worker, n * 4)
    count_buf = _output_buffer(worker, 4)
    params_buf = _uniform_buffer(worker, n)

    bind_group = worker.create_bind_group(
        pipeline,
        0,
        [
            {"binding": 0, "resource": {"buffer": val_buf}},
            {"binding": 1, "resource": {"buffer": mask_buf}},
            {"binding": 2, "resource": {"buffer": out_buf}},
            {"binding": 3, "resource": {"buffer": count_buf}},
            {"binding": 4, "resource": {"buffer": params_buf}},
        ],
    )

    worker.dispatch_compute(pipeline, [bind_group], (1, 1, 1))

    count_raw = worker.read_buffer(count_buf, 4)
    count = int(np.frombuffer(count_raw, dtype=np.uint32)[0])
    if count > 0:
        val_raw = worker.read_buffer(out_buf, count * 4)
        result = np.frombuffer(val_raw, dtype=np.int32).copy()
    else:
        result = np.array([], dtype=np.int32)
    return result, count


def dispatch_segmented_scan(
    worker: GPUWorker, values: np.ndarray, segment_ids: np.ndarray
) -> np.ndarray:
    """Run segmented exclusive prefix sum on GPU. Returns i32 scan result.

    values: i32 array, segment_ids: i32 array (non-decreasing). Max 256 elements.
    """
    n = len(values)
    if n == 0:
        return np.array([], dtype=np.int32)
    if n > MAX_WORKGROUP_ELEMENTS:
        raise ValueError(
            f"segmented_scan: {n} elements exceeds workgroup max {MAX_WORKGROUP_ELEMENTS}"
        )

    values_i32 = values.astype(np.int32)
    seg_i32 = segment_ids.astype(np.int32)
    wgsl = load_shader("segmented_scan")
    pipeline = worker.create_compute_pipeline(wgsl, "main")

    val_buf = _storage_buffer_from_numpy(worker, values_i32)
    seg_buf = _storage_buffer_from_numpy(worker, seg_i32)
    out_buf = _output_buffer(worker, n * 4)
    params_buf = _uniform_buffer(worker, n)

    bind_group = worker.create_bind_group(
        pipeline,
        0,
        [
            {"binding": 0, "resource": {"buffer": val_buf}},
            {"binding": 1, "resource": {"buffer": seg_buf}},
            {"binding": 2, "resource": {"buffer": out_buf}},
            {"binding": 3, "resource": {"buffer": params_buf}},
        ],
    )

    worker.dispatch_compute(pipeline, [bind_group], (1, 1, 1))
    raw = worker.read_buffer(out_buf, n * 4)
    return np.frombuffer(raw, dtype=np.int32).copy()
