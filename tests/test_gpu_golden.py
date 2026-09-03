"""Golden conformance vectors for GPU compute primitives.

Each test encodes a fixed input and its expected output. The CPU
reference implementation must produce these exact results. When GPU
shaders are added, they run the same vectors and assert byte-identical
output (first-mismatch reporting).
"""

import numpy as np
import pytest

from chitin.gpu.primitives import (
    compact,
    prefix_sum_exclusive,
    prefix_sum_inclusive,
    reduce_sum,
    segmented_scan,
    stable_sort_by_key,
    unique_sorted,
)


# Fixed seed for reproducible "realistic" vectors
_RNG = np.random.default_rng(seed=0xDEAD_BEEF)

# --- Golden vectors ---

GOLDEN_SCAN_INPUT = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int32)
GOLDEN_SCAN_EXCLUSIVE = np.array([0, 3, 4, 8, 9, 14, 23, 25], dtype=np.int64)
GOLDEN_SCAN_INCLUSIVE = np.array([3, 4, 8, 9, 14, 23, 25, 31], dtype=np.int64)
GOLDEN_SCAN_TOTAL = 31

GOLDEN_COMPACT_VALUES = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int32)
GOLDEN_COMPACT_MASK = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int32)
GOLDEN_COMPACT_RESULT = np.array([10, 30, 40, 70], dtype=np.int32)
GOLDEN_COMPACT_COUNT = 4

GOLDEN_SORT_KEYS = np.array([5, 3, 8, 3, 1, 7], dtype=np.int32)
GOLDEN_SORT_VALUES = np.array([50, 30, 80, 31, 10, 70], dtype=np.int32)
GOLDEN_SORT_KEYS_OUT = np.array([1, 3, 3, 5, 7, 8], dtype=np.int32)
GOLDEN_SORT_VALUES_OUT = np.array([10, 30, 31, 50, 70, 80], dtype=np.int32)

GOLDEN_UNIQUE_INPUT = np.array([1, 1, 2, 2, 2, 3, 5, 5, 8], dtype=np.int32)
GOLDEN_UNIQUE_RESULT = np.array([1, 2, 3, 5, 8], dtype=np.int32)
GOLDEN_UNIQUE_COUNT = 5

GOLDEN_REDUCE_INPUT = np.array(
    [1.5, 2.25, 3.125, 4.0625, 5.03125, 6.015625, 7.0078125, 8.00390625],
    dtype=np.float64,
)
GOLDEN_REDUCE_SUM = 36.99609375

GOLDEN_SEG_VALUES = np.array([1, 2, 3, 10, 20, 30, 100], dtype=np.int32)
GOLDEN_SEG_IDS = np.array([0, 0, 0, 1, 1, 1, 2], dtype=np.int32)
GOLDEN_SEG_RESULT = np.array([0, 1, 3, 0, 10, 30, 0], dtype=np.int64)

# Larger vector for stress/determinism
GOLDEN_LARGE_N = 10000
GOLDEN_LARGE_INPUT = _RNG.integers(0, 1000, size=GOLDEN_LARGE_N, dtype=np.int32)
GOLDEN_LARGE_SCAN_TOTAL = int(np.sum(GOLDEN_LARGE_INPUT, dtype=np.int64))


class TestGoldenScan:
    def test_exclusive(self):
        result = prefix_sum_exclusive(GOLDEN_SCAN_INPUT)
        np.testing.assert_array_equal(result, GOLDEN_SCAN_EXCLUSIVE)

    def test_inclusive(self):
        scan = prefix_sum_inclusive(GOLDEN_SCAN_INPUT)
        np.testing.assert_array_equal(scan.inclusive, GOLDEN_SCAN_INCLUSIVE)
        assert scan.total == GOLDEN_SCAN_TOTAL

    def test_large_total(self):
        scan = prefix_sum_inclusive(GOLDEN_LARGE_INPUT)
        assert scan.total == GOLDEN_LARGE_SCAN_TOTAL


class TestGoldenCompact:
    def test_values_and_count(self):
        result, count = compact(GOLDEN_COMPACT_VALUES, GOLDEN_COMPACT_MASK)
        np.testing.assert_array_equal(result, GOLDEN_COMPACT_RESULT)
        assert count == GOLDEN_COMPACT_COUNT


class TestGoldenSort:
    def test_keys_and_values(self):
        sk, sv = stable_sort_by_key(GOLDEN_SORT_KEYS, GOLDEN_SORT_VALUES)
        np.testing.assert_array_equal(sk, GOLDEN_SORT_KEYS_OUT)
        np.testing.assert_array_equal(sv, GOLDEN_SORT_VALUES_OUT)


class TestGoldenUnique:
    def test_values_and_count(self):
        result, count = unique_sorted(GOLDEN_UNIQUE_INPUT)
        np.testing.assert_array_equal(result, GOLDEN_UNIQUE_RESULT)
        assert count == GOLDEN_UNIQUE_COUNT


class TestGoldenReduce:
    def test_sum_exact(self):
        result = reduce_sum(GOLDEN_REDUCE_INPUT)
        assert result.value == GOLDEN_REDUCE_SUM
        assert result.count == 8


class TestGoldenSegmented:
    def test_segmented_scan(self):
        result = segmented_scan(GOLDEN_SEG_VALUES, GOLDEN_SEG_IDS)
        np.testing.assert_array_equal(result, GOLDEN_SEG_RESULT)


class TestFirstMismatchReport:
    """Verify that when a mismatch occurs, we get useful diagnostics."""

    def test_mismatch_reporting(self):
        result = prefix_sum_exclusive(GOLDEN_SCAN_INPUT)
        wrong = GOLDEN_SCAN_EXCLUSIVE.copy()
        wrong[3] = 999
        with pytest.raises(AssertionError) as exc_info:
            np.testing.assert_array_equal(result, wrong)
        assert "3" in str(exc_info.value) or "index" in str(exc_info.value).lower()
