"""CPU reference implementations for the WGSL kernels in this package.

These functions deliberately expose the same fixed-width arithmetic and
accumulation order as their GPU equivalents. CPU-only helpers belong with the
host algorithm, not in this GPU conformance surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_WORKGROUP_ELEMENTS = 256


def _i32_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int32)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return array


def _check_workgroup_size(values: np.ndarray, operation: str) -> None:
    if len(values) > MAX_WORKGROUP_ELEMENTS:
        raise ValueError(
            f"{operation}: {len(values)} elements exceeds workgroup max "
            f"{MAX_WORKGROUP_ELEMENTS}"
        )


def prefix_sum_exclusive(values: np.ndarray) -> np.ndarray:
    """WGSL-equivalent exclusive ``i32`` scan for one 256-lane workgroup."""
    values_i32 = _i32_vector(values, "values")
    _check_workgroup_size(values_i32, "prefix_sum")
    result = np.zeros(len(values_i32), dtype=np.int32)
    if len(values_i32) > 1:
        result[1:] = np.cumsum(values_i32[:-1], dtype=np.int32)
    return result


def compact(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    """WGSL-equivalent stable ``i32`` stream compaction."""
    values_i32 = _i32_vector(values, "values")
    mask_i32 = _i32_vector(mask, "mask")
    _check_workgroup_size(values_i32, "compact")
    if len(mask_i32) != len(values_i32):
        raise ValueError("compact requires values and mask with equal lengths")
    selected = values_i32[mask_i32 != 0]
    return selected, len(selected)


@dataclass(frozen=True, slots=True)
class ReductionResult:
    value: float
    count: int


def reduce_sum(values: np.ndarray) -> ReductionResult:
    """WGSL-equivalent fixed 256-lane ``f32`` reduction tree."""
    values_f32 = np.asarray(values, dtype=np.float32)
    if values_f32.ndim != 1:
        raise ValueError("values must be a one-dimensional array")
    _check_workgroup_size(values_f32, "reduce_sum")
    if len(values_f32) == 0:
        return ReductionResult(0.0, 0)

    shared = np.zeros(MAX_WORKGROUP_ELEMENTS, dtype=np.float32)
    shared[: len(values_f32)] = values_f32
    for stride in (128, 64, 32, 16, 8, 4, 2, 1):
        shared[:stride] = shared[:stride] + shared[stride : 2 * stride]
    return ReductionResult(float(shared[0]), len(values_f32))


def segmented_scan(values: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    """WGSL-equivalent exclusive ``i32`` scan within sorted segments."""
    values_i32 = _i32_vector(values, "values")
    segment_ids_i32 = _i32_vector(segment_ids, "segment_ids")
    _check_workgroup_size(values_i32, "segmented_scan")
    if len(segment_ids_i32) != len(values_i32):
        raise ValueError(
            "segmented_scan requires values and segment_ids with equal lengths"
        )
    if len(values_i32) == 0:
        return np.array([], dtype=np.int32)
    if np.any(segment_ids_i32[1:] < segment_ids_i32[:-1]):
        raise ValueError("segmented_scan requires non-decreasing segment_ids")

    result = np.zeros(len(values_i32), dtype=np.int32)
    current_segment = segment_ids_i32[0]
    accumulator = np.int32(0)
    for index, value in enumerate(values_i32):
        if segment_ids_i32[index] != current_segment:
            current_segment = segment_ids_i32[index]
            accumulator = np.int32(0)
        result[index] = accumulator
        accumulator = np.add(accumulator, value, dtype=np.int32)
    return result
