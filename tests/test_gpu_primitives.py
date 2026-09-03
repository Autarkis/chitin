"""Tests for deterministic bounded compute primitives."""

import numpy as np
import pytest

from chitin.gpu.primitives import (
    compact,
    prefix_sum_exclusive,
    prefix_sum_inclusive,
    reduce_max,
    reduce_min,
    reduce_sum,
    segmented_scan,
    stable_sort_by_key,
    unique_sorted,
    unique_with_counts,
)


class TestPrefixSum:
    def test_exclusive_basic(self):
        arr = np.array([1, 2, 3, 4], dtype=np.int32)
        result = prefix_sum_exclusive(arr)
        np.testing.assert_array_equal(result, [0, 1, 3, 6])

    def test_exclusive_empty(self):
        result = prefix_sum_exclusive(np.array([], dtype=np.int32))
        assert len(result) == 0

    def test_exclusive_single(self):
        result = prefix_sum_exclusive(np.array([7], dtype=np.int32))
        np.testing.assert_array_equal(result, [0])

    def test_inclusive_basic(self):
        arr = np.array([1, 2, 3, 4], dtype=np.int32)
        result = prefix_sum_inclusive(arr)
        np.testing.assert_array_equal(result.inclusive, [1, 3, 6, 10])
        assert result.total == 10

    def test_inclusive_empty(self):
        result = prefix_sum_inclusive(np.array([], dtype=np.int32))
        assert result.total == 0

    def test_deterministic(self):
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 100, size=1000, dtype=np.int32)
        r1 = prefix_sum_exclusive(arr)
        r2 = prefix_sum_exclusive(arr)
        np.testing.assert_array_equal(r1, r2)


class TestCompact:
    def test_basic(self):
        values = np.array([10, 20, 30, 40, 50])
        mask = np.array([1, 0, 1, 0, 1])
        result, count = compact(values, mask)
        np.testing.assert_array_equal(result, [10, 30, 50])
        assert count == 3

    def test_all_selected(self):
        values = np.array([1, 2, 3])
        mask = np.ones(3, dtype=np.int32)
        result, count = compact(values, mask)
        np.testing.assert_array_equal(result, [1, 2, 3])
        assert count == 3

    def test_none_selected(self):
        values = np.array([1, 2, 3])
        mask = np.zeros(3, dtype=np.int32)
        result, count = compact(values, mask)
        assert count == 0

    def test_empty(self):
        result, count = compact(np.array([]), np.array([]))
        assert count == 0

    def test_preserves_order(self):
        values = np.array([5, 3, 1, 4, 2])
        mask = np.array([1, 0, 1, 1, 0])
        result, _ = compact(values, mask)
        np.testing.assert_array_equal(result, [5, 1, 4])


class TestStableSort:
    def test_basic(self):
        keys = np.array([3, 1, 2])
        vals = np.array([30, 10, 20])
        sk, sv = stable_sort_by_key(keys, vals)
        np.testing.assert_array_equal(sk, [1, 2, 3])
        np.testing.assert_array_equal(sv, [10, 20, 30])

    def test_stability_on_ties(self):
        keys = np.array([2, 1, 2, 1])
        vals = np.array([0, 1, 2, 3])
        _, sv = stable_sort_by_key(keys, vals)
        np.testing.assert_array_equal(sv, [1, 3, 0, 2])

    def test_empty(self):
        k = np.array([], dtype=np.int32)
        v = np.array([], dtype=np.int32)
        sk, sv = stable_sort_by_key(k, v)
        assert len(sk) == 0
        assert len(sv) == 0

    def test_already_sorted(self):
        keys = np.array([1, 2, 3])
        vals = np.array([10, 20, 30])
        _, sv = stable_sort_by_key(keys, vals)
        np.testing.assert_array_equal(sv, [10, 20, 30])


class TestUnique:
    def test_sorted_basic(self):
        arr = np.array([1, 1, 2, 3, 3, 3])
        result, count = unique_sorted(arr)
        np.testing.assert_array_equal(result, [1, 2, 3])
        assert count == 3

    def test_empty(self):
        result, count = unique_sorted(np.array([], dtype=np.int32))
        assert count == 0

    def test_with_counts(self):
        arr = np.array([1, 1, 2, 3, 3, 3])
        uniq, counts = unique_with_counts(arr)
        np.testing.assert_array_equal(uniq, [1, 2, 3])
        np.testing.assert_array_equal(counts, [2, 1, 3])


class TestReduction:
    def test_sum_basic(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = reduce_sum(arr)
        assert result.value == pytest.approx(10.0)
        assert result.count == 4

    def test_sum_empty(self):
        result = reduce_sum(np.array([]))
        assert result.value == 0.0
        assert result.count == 0

    def test_sum_deterministic(self):
        rng = np.random.default_rng(42)
        arr = rng.random(10000).astype(np.float32)
        r1 = reduce_sum(arr)
        r2 = reduce_sum(arr)
        assert r1.value == r2.value

    def test_min(self):
        arr = np.array([3.0, 1.0, 4.0, 1.5])
        result = reduce_min(arr)
        assert result.value == pytest.approx(1.0)

    def test_min_empty(self):
        result = reduce_min(np.array([]))
        assert result.value == float("inf")

    def test_max(self):
        arr = np.array([3.0, 1.0, 4.0, 1.5])
        result = reduce_max(arr)
        assert result.value == pytest.approx(4.0)

    def test_max_empty(self):
        result = reduce_max(np.array([]))
        assert result.value == float("-inf")


class TestSegmentedScan:
    def test_basic(self):
        values = np.array([1, 2, 3, 10, 20], dtype=np.int32)
        seg_ids = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        result = segmented_scan(values, seg_ids)
        np.testing.assert_array_equal(result, [0, 1, 3, 0, 10])

    def test_single_segment(self):
        values = np.array([1, 2, 3], dtype=np.int32)
        seg_ids = np.array([0, 0, 0], dtype=np.int32)
        result = segmented_scan(values, seg_ids)
        np.testing.assert_array_equal(result, [0, 1, 3])

    def test_each_own_segment(self):
        values = np.array([5, 10, 15], dtype=np.int32)
        seg_ids = np.array([0, 1, 2], dtype=np.int32)
        result = segmented_scan(values, seg_ids)
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_empty(self):
        result = segmented_scan(
            np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        )
        assert len(result) == 0
