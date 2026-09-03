"""Typed one-workgroup WGSL kernel dispatch.

Kernel metadata owns input validation, shader selection, and binding order so
individual public helpers only describe their result contract.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from chitin.gpu.primitives import MAX_WORKGROUP_ELEMENTS
from chitin.gpu.shaders import load_shader
from chitin.gpu.worker import GPUWorker

_STORAGE = 0x80
_COPY_SRC = 0x04
_COPY_DST = 0x08
_UNIFORM = 0x40


@dataclass(frozen=True, slots=True)
class KernelSpec:
    name: str
    input_names: tuple[str, ...]
    input_dtype: np.dtype
    output_sizes: Callable[[int], tuple[int, ...]]

    def validate(self, arrays: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        if len(arrays) != len(self.input_names):
            raise ValueError(f"{self.name} received an invalid number of inputs")
        normalized = tuple(
            _vector(array, dtype=self.input_dtype, name=name)
            for name, array in zip(self.input_names, arrays, strict=True)
        )
        count = len(normalized[0])
        if count > MAX_WORKGROUP_ELEMENTS:
            raise ValueError(
                f"{self.name}: {count} elements exceeds workgroup max "
                f"{MAX_WORKGROUP_ELEMENTS}"
            )
        if any(len(array) != count for array in normalized[1:]):
            names = ", ".join(self.input_names)
            raise ValueError(f"{self.name} requires equal-length {names} arrays")
        return normalized


PREFIX_SUM = KernelSpec(
    "prefix_sum", ("values",), np.dtype(np.int32), lambda n: ((n + 1) * 4,)
)
REDUCE_SUM = KernelSpec(
    "reduce_sum", ("values",), np.dtype(np.float32), lambda _n: (4,)
)
COMPACT = KernelSpec(
    "compact", ("values", "mask"), np.dtype(np.int32), lambda n: (n * 4, 4)
)
SEGMENTED_SCAN = KernelSpec(
    "segmented_scan", ("values", "segment_ids"), np.dtype(np.int32), lambda n: (n * 4,)
)


def _vector(values: np.ndarray, *, dtype: np.dtype, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return array


def _buffer_from_numpy(worker: GPUWorker, array: np.ndarray) -> object:
    data = array.tobytes()
    buffer = worker.create_buffer(max(len(data), 4), _STORAGE | _COPY_DST)
    if data:
        worker.write_buffer(buffer, data)
    return buffer


def _output_buffer(worker: GPUWorker, size: int) -> object:
    return worker.create_buffer(max(size, 4), _STORAGE | _COPY_SRC)


def _uniform_buffer(worker: GPUWorker, count: int) -> object:
    buffer = worker.create_buffer(16, _UNIFORM | _COPY_DST)
    worker.write_buffer(buffer, struct.pack("<IIII", count, 0, 0, 0))
    return buffer


def _run(worker: GPUWorker, spec: KernelSpec, *arrays: np.ndarray) -> tuple[bytes, ...]:
    inputs = spec.validate(arrays)
    count = len(inputs[0])
    if count == 0:
        raise ValueError("_run only accepts non-empty input")

    operation = worker.begin_operation()
    pipeline = worker.create_compute_pipeline(load_shader(spec.name))
    input_buffers = [_buffer_from_numpy(worker, array) for array in inputs]
    output_sizes = spec.output_sizes(count)
    output_buffers = [_output_buffer(worker, size) for size in output_sizes]
    params_buffer = _uniform_buffer(worker, count)
    buffers = [*input_buffers, *output_buffers, params_buffer]
    bind_group = worker.create_bind_group(
        pipeline,
        0,
        [
            {"binding": index, "resource": {"buffer": buffer}}
            for index, buffer in enumerate(buffers)
        ],
    )
    worker.dispatch_compute(pipeline, [bind_group], (1, 1, 1))
    worker.check_operation(operation)
    result = tuple(
        worker.read_buffer(buffer, size)
        for buffer, size in zip(output_buffers, output_sizes, strict=True)
    )
    worker.check_operation(operation)
    return result


def dispatch_prefix_sum_exclusive(
    worker: GPUWorker, values: np.ndarray
) -> tuple[np.ndarray, int]:
    """Run the bounded exclusive ``i32`` scan kernel."""
    values = PREFIX_SUM.validate((values,))[0]
    if len(values) == 0:
        return np.array([], dtype=np.int32), 0
    (raw,) = _run(worker, PREFIX_SUM, values)
    result = np.frombuffer(raw, dtype=np.int32)
    return result[:-1].copy(), int(result[-1])


def dispatch_reduce_sum(worker: GPUWorker, values: np.ndarray) -> float:
    """Run the bounded fixed-tree ``f32`` reduction kernel."""
    values = REDUCE_SUM.validate((values,))[0]
    if len(values) == 0:
        return 0.0
    (raw,) = _run(worker, REDUCE_SUM, values)
    return float(np.frombuffer(raw, dtype=np.float32)[0])


def dispatch_compact(
    worker: GPUWorker, values: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, int]:
    """Run stable ``i32`` stream compaction."""
    values, mask = COMPACT.validate((values, mask))
    if len(values) == 0:
        return np.array([], dtype=np.int32), 0
    values_raw, count_raw = _run(worker, COMPACT, values, mask)
    count = int(np.frombuffer(count_raw, dtype=np.uint32)[0])
    return np.frombuffer(values_raw, dtype=np.int32, count=count).copy(), count


def dispatch_segmented_scan(
    worker: GPUWorker, values: np.ndarray, segment_ids: np.ndarray
) -> np.ndarray:
    """Run exclusive ``i32`` scan within non-decreasing segments."""
    values, segment_ids = SEGMENTED_SCAN.validate((values, segment_ids))
    if np.any(segment_ids[1:] < segment_ids[:-1]):
        raise ValueError("segmented_scan requires non-decreasing segment_ids")
    if len(values) == 0:
        return np.array([], dtype=np.int32)
    (raw,) = _run(worker, SEGMENTED_SCAN, values, segment_ids)
    return np.frombuffer(raw, dtype=np.int32).copy()
