"""Tests for CPU references that mirror shipped GPU kernels."""

import numpy as np
import pytest

from chitin.gpu.primitives import (
    compact,
    prefix_sum_exclusive,
    reduce_sum,
    segmented_scan,
)


class TestPrefixSum:
    def test_exclusive_basic(self):
        result = prefix_sum_exclusive(np.array([1, 2, 3, 4], dtype=np.int32))
        np.testing.assert_array_equal(result, np.array([0, 1, 3, 6], dtype=np.int32))

    def test_i32_overflow_matches_wgsl(self):
        result = prefix_sum_exclusive(
            np.array([np.iinfo(np.int32).max, 1, 2], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            result,
            np.array(
                [0, np.iinfo(np.int32).max, np.iinfo(np.int32).min], dtype=np.int32
            ),
        )

    def test_rejects_oversized_workgroup(self):
        with pytest.raises(ValueError, match="workgroup max"):
            prefix_sum_exclusive(np.ones(257, dtype=np.int32))


class TestCompact:
    def test_preserves_order(self):
        result, count = compact(
            np.array([5, 3, 1, 4, 2], dtype=np.int32),
            np.array([1, 0, 1, 1, 0], dtype=np.int32),
        )
        np.testing.assert_array_equal(result, np.array([5, 1, 4], dtype=np.int32))
        assert count == 3

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="equal lengths"):
            compact(np.array([1, 2], dtype=np.int32), np.array([1], dtype=np.int32))


class TestReduction:
    def test_sum_basic(self):
        result = reduce_sum(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        assert result.value == 10.0
        assert result.count == 4

    def test_f32_tree_contract(self):
        values = np.array([1e20, 1.0, -1e20], dtype=np.float32)
        assert reduce_sum(values).value == 1.0


class TestSegmentedScan:
    def test_basic(self):
        result = segmented_scan(
            np.array([1, 2, 3, 10, 20], dtype=np.int32),
            np.array([0, 0, 0, 1, 1], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            result, np.array([0, 1, 3, 0, 10], dtype=np.int32)
        )

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="equal lengths"):
            segmented_scan(np.array([1], dtype=np.int32), np.array([], dtype=np.int32))

    def test_rejects_unsorted_segment_ids(self):
        with pytest.raises(ValueError, match="non-decreasing"):
            segmented_scan(
                np.array([1, 2], dtype=np.int32), np.array([1, 0], dtype=np.int32)
            )
