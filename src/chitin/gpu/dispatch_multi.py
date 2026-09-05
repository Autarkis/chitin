"""Unbounded multi-workgroup WGSL kernel dispatch.

Tiles arbitrary-sized input into 256-element blocks, scans or reduces each
tile independently, then combines the per-tile results with a second pass
that reuses the existing single-workgroup kernels from `chitin.gpu.dispatch`
(bounded to 256 elements, which the per-tile results always satisfy up to the
two-level cap enforced below).
"""

from __future__ import annotations

import math
import struct

import numpy as np

from chitin.gpu.shaders import load_shader
from chitin.gpu.worker import GPUWorker

_STORAGE = 0x80
_COPY_SRC = 0x04
_COPY_DST = 0x08
_UNIFORM = 0x40

_TILE = 256


def _buffer_from_numpy(worker: GPUWorker, array: np.ndarray) -> object:
    data = array.tobytes()
    buffer = worker.create_buffer(max(len(data), 4), _STORAGE | _COPY_DST)
    if data:
        worker.write_buffer(buffer, data)
    return buffer


def _rw_buffer(worker: GPUWorker, size: int) -> object:
    return worker.create_buffer(max(size, 4), _STORAGE | _COPY_SRC | _COPY_DST)


def _uniform_buffer(worker: GPUWorker, count: int) -> object:
    buffer = worker.create_buffer(16, _UNIFORM | _COPY_DST)
    worker.write_buffer(buffer, struct.pack("<IIII", count, 0, 0, 0))
    return buffer


def _dispatch_pass(
    worker: GPUWorker,
    shader_name: str,
    buffers: list[object],
    workgroups: tuple[int, int, int],
) -> None:
    """Build the named kernel's pipeline, bind buffers 0..N-1, and dispatch it."""
    operation = worker.begin_operation()
    pipeline = worker.create_compute_pipeline(load_shader(shader_name))
    bind_group = worker.create_bind_group(
        pipeline,
        0,
        [
            {"binding": index, "resource": {"buffer": buffer}}
            for index, buffer in enumerate(buffers)
        ],
    )
    worker.dispatch_compute(pipeline, [bind_group], workgroups)
    worker.check_operation(operation)


def dispatch_prefix_sum_exclusive_multi(
    worker: GPUWorker, values: np.ndarray
) -> tuple[np.ndarray, int]:
    """Run the unbounded exclusive ``i32`` scan via tile-scan, block-scan, propagate."""
    values = np.asarray(values, dtype=np.int32).flatten()
    n = len(values)
    if n == 0:
        return np.array([], dtype=np.int32), 0

    num_blocks = math.ceil(n / _TILE)
    if num_blocks > _TILE:
        raise ValueError(
            f"dispatch_prefix_sum_exclusive_multi: {n} elements need "
            f"{num_blocks} tiles, exceeding the two-level scan cap of "
            f"{_TILE * _TILE} elements"
        )

    input_buf = _buffer_from_numpy(worker, values)
    output_buf = _rw_buffer(worker, n * 4)
    block_sums_buf = _rw_buffer(worker, num_blocks * 4)
    params_buf = _uniform_buffer(worker, n)

    _dispatch_pass(
        worker,
        "prefix_sum_local",
        [input_buf, output_buf, block_sums_buf, params_buf],
        (num_blocks, 1, 1),
    )

    if num_blocks == 1:
        # A single tile has no block offset to propagate: its tile total,
        # written verbatim by prefix_sum_local into block_sums[0], is already
        # the grand total.
        operation = worker.begin_operation()
        result_raw = worker.read_buffer(output_buf, n * 4)
        total_raw = worker.read_buffer(block_sums_buf, 4)
        worker.check_operation(operation)
        result = np.frombuffer(result_raw, dtype=np.int32).copy()
        total = int(np.frombuffer(total_raw, dtype=np.int32)[0])
        return result, total

    blocks_params_buf = _uniform_buffer(worker, num_blocks)
    scanned_blocks_buf = _rw_buffer(worker, (num_blocks + 1) * 4)
    _dispatch_pass(
        worker,
        "prefix_sum",
        [block_sums_buf, scanned_blocks_buf, blocks_params_buf],
        (1, 1, 1),
    )

    _dispatch_pass(
        worker,
        "prefix_sum_propagate",
        [output_buf, scanned_blocks_buf, params_buf],
        (num_blocks, 1, 1),
    )

    operation = worker.begin_operation()
    result_raw = worker.read_buffer(output_buf, n * 4)
    # prefix_sum.wgsl writes one element past the requested count with the
    # scan's total, so scanned_blocks_buf[num_blocks] is the true grand total
    # of all block sums (equivalently, of all n input elements) -- read it
    # from that buffer rather than reconstructing it from the tiled result,
    # which only holds per-tile exclusive offsets.
    total_raw = worker.read_buffer(scanned_blocks_buf, (num_blocks + 1) * 4)
    worker.check_operation(operation)
    result = np.frombuffer(result_raw, dtype=np.int32).copy()
    total = int(np.frombuffer(total_raw, dtype=np.int32)[-1])
    return result, total


def dispatch_compact_multi(
    worker: GPUWorker, values: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, int]:
    """Run unbounded stable ``i32`` stream compaction: GPU scan, CPU scatter."""
    values = np.asarray(values, dtype=np.int32).flatten()
    mask = np.asarray(mask, dtype=np.int32).flatten()
    if len(values) != len(mask):
        raise ValueError("dispatch_compact_multi requires equal-length values and mask")
    if len(values) == 0:
        return np.array([], dtype=np.int32), 0

    mask_binary = mask != 0
    offsets, total = dispatch_prefix_sum_exclusive_multi(
        worker, mask_binary.astype(np.int32)
    )
    if total == 0:
        return np.array([], dtype=np.int32), 0

    # The scatter itself stays on the CPU: at these sizes the scan above is
    # the bottleneck, and a GPU scatter kernel is a deliberately deferred
    # optimization, not an oversight.
    selected = np.empty(total, dtype=np.int32)
    for i in range(len(values)):
        if mask_binary[i]:
            selected[offsets[i]] = values[i]
    return selected, total


def dispatch_reduce_sum_multi(worker: GPUWorker, values: np.ndarray) -> float:
    """Run the unbounded tree-reduction ``f32`` sum via block-partials, then reduce."""
    values = np.asarray(values, dtype=np.float32).flatten()
    n = len(values)
    if n == 0:
        return 0.0

    num_blocks = math.ceil(n / _TILE)
    if num_blocks == 1:
        from chitin.gpu.dispatch import dispatch_reduce_sum

        return dispatch_reduce_sum(worker, values)

    if num_blocks > _TILE:
        raise ValueError(
            f"dispatch_reduce_sum_multi: {n} elements need {num_blocks} "
            f"block partials, exceeding the single-workgroup partial-reduce "
            f"cap of {_TILE}"
        )

    input_buf = _buffer_from_numpy(worker, values)
    partials_buf = _rw_buffer(worker, num_blocks * 4)
    params_buf = _uniform_buffer(worker, n)

    _dispatch_pass(
        worker,
        "reduce_sum_multi",
        [input_buf, partials_buf, params_buf],
        (num_blocks, 1, 1),
    )

    final_buf = _rw_buffer(worker, 4)
    partials_params_buf = _uniform_buffer(worker, num_blocks)
    _dispatch_pass(
        worker,
        "reduce_sum",
        [partials_buf, final_buf, partials_params_buf],
        (1, 1, 1),
    )

    operation = worker.begin_operation()
    raw = worker.read_buffer(final_buf, 4)
    worker.check_operation(operation)
    return float(np.frombuffer(raw, dtype=np.float32)[0])
