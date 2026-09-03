"""Stable CPU vectors for every shipped WGSL kernel."""

import numpy as np

from chitin.gpu.primitives import (
    compact,
    prefix_sum_exclusive,
    reduce_sum,
    segmented_scan,
)


def test_prefix_sum_golden_vector():
    result = prefix_sum_exclusive(np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int32))
    np.testing.assert_array_equal(
        result, np.array([0, 3, 4, 8, 9, 14, 23, 25], dtype=np.int32)
    )


def test_compact_golden_vector():
    result, count = compact(
        np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int32),
        np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(result, np.array([10, 30, 40, 70], dtype=np.int32))
    assert count == 4


def test_reduce_sum_golden_vector():
    result = reduce_sum(
        np.array(
            [1.5, 2.25, 3.125, 4.0625, 5.03125, 6.015625, 7.0078125, 8.00390625],
            dtype=np.float32,
        )
    )
    assert result.value == 36.99609375
    assert result.count == 8


def test_segmented_scan_golden_vector():
    result = segmented_scan(
        np.array([1, 2, 3, 10, 20, 30, 100], dtype=np.int32),
        np.array([0, 0, 0, 1, 1, 1, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result, np.array([0, 1, 3, 0, 10, 30, 0], dtype=np.int32)
    )
