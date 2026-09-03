"""Deterministic bounded compute primitives.

CPU reference implementations matching WGSL shader behavior. Each
primitive operates on bounded arrays and produces bit-identical results
to its GPU counterpart (verified by golden conformance tests).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ScanResult:
    inclusive: np.ndarray
    total: int


def prefix_sum_exclusive(values: np.ndarray) -> np.ndarray:
    """Exclusive prefix sum (Blelloch-style). Deterministic."""
    if len(values) == 0:
        return np.array([], dtype=values.dtype)
    result = np.zeros(len(values), dtype=np.int64)
    acc = np.int64(0)
    for i in range(len(values)):
        result[i] = acc
        acc += np.int64(values[i])
    return result


def prefix_sum_inclusive(values: np.ndarray) -> ScanResult:
    """Inclusive prefix sum. Returns partial sums and total."""
    if len(values) == 0:
        return ScanResult(np.array([], dtype=np.int64), 0)
    result = np.cumsum(values, dtype=np.int64)
    return ScanResult(result, int(result[-1]))


def compact(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Stream compaction: keep elements where mask is nonzero.

    Returns (compacted_array, count). Preserves relative order
    (stability required for determinism).
    """
    if len(values) == 0:
        return np.array([], dtype=values.dtype), 0
    selected = values[mask.astype(bool)]
    return selected, len(selected)


def stable_sort_by_key(
    keys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Stable sort by keys, carrying values. Deterministic ordering.

    Ties are broken by original index (stability guarantee).
    """
    if len(keys) == 0:
        return keys.copy(), values.copy()
    order = np.argsort(keys, kind="stable")
    return keys[order], values[order]


def unique_sorted(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove duplicates from a sorted array. Returns (unique, count)."""
    if len(values) == 0:
        return np.array([], dtype=values.dtype), 0
    result = np.unique(values)
    return result, len(result)


def unique_with_counts(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique values with their counts from a sorted array."""
    if len(values) == 0:
        return np.array([], dtype=values.dtype), np.array([], dtype=np.int64)
    uniq, counts = np.unique(values, return_counts=True)
    return uniq, counts.astype(np.int64)


@dataclass(frozen=True, slots=True)
class ReductionResult:
    value: float
    count: int


def reduce_sum(values: np.ndarray) -> ReductionResult:
    """Fixed-tree sum reduction. Deterministic accumulation order.

    Uses pairwise summation (binary tree) for reproducibility.
    Matches the GPU reduction tree structure.
    """
    if len(values) == 0:
        return ReductionResult(0.0, 0)
    total = float(_pairwise_sum(values.astype(np.float64)))
    return ReductionResult(total, len(values))


def reduce_min(values: np.ndarray) -> ReductionResult:
    """Tree-based minimum reduction."""
    if len(values) == 0:
        return ReductionResult(float("inf"), 0)
    return ReductionResult(float(np.min(values)), len(values))


def reduce_max(values: np.ndarray) -> ReductionResult:
    """Tree-based maximum reduction."""
    if len(values) == 0:
        return ReductionResult(float("-inf"), 0)
    return ReductionResult(float(np.max(values)), len(values))


def _pairwise_sum(arr: np.ndarray) -> np.float64:
    """Binary-tree pairwise summation for deterministic accumulation."""
    n = len(arr)
    if n == 0:
        return np.float64(0.0)
    if n == 1:
        return np.float64(arr[0])
    if n <= 8:
        s = np.float64(0.0)
        for v in arr:
            s += np.float64(v)
        return s
    mid = n // 2
    return _pairwise_sum(arr[:mid]) + _pairwise_sum(arr[mid:])


def segmented_scan(values: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    """Exclusive prefix sum within segments defined by segment_ids.

    Each segment restarts the accumulator. segment_ids must be
    non-decreasing (sorted).
    """
    if len(values) == 0:
        return np.array([], dtype=np.int64)
    result = np.zeros(len(values), dtype=np.int64)
    acc = np.int64(0)
    current_seg = segment_ids[0]
    for i in range(len(values)):
        if segment_ids[i] != current_seg:
            acc = np.int64(0)
            current_seg = segment_ids[i]
        result[i] = acc
        acc += np.int64(values[i])
    return result
