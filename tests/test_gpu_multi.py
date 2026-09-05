"""Multi-workgroup GPU dispatch tests — verify multi-tile primitives
against CPU reference at sizes exceeding the 256-element workgroup limit."""

import numpy as np
import pytest

from chitin.gpu.primitives import (
    compact_multi,
    prefix_sum_exclusive_multi,
    reduce_sum_multi,
)
from chitin.gpu.worker import GPUWorker


@pytest.fixture(scope="module")
def worker():
    if not GPUWorker.available():
        pytest.skip("No GPU adapter available")
    with GPUWorker() as w:
        yield w


class TestCPUMultiPrefixSum:
    """CPU reference correctness (no GPU needed)."""

    def test_small(self):
        values = np.array([3, 1, 4, 1, 5], dtype=np.int32)
        result, total = prefix_sum_exclusive_multi(values)
        np.testing.assert_array_equal(result, [0, 3, 4, 8, 9])
        assert total == 14

    def test_empty(self):
        result, total = prefix_sum_exclusive_multi(np.array([], dtype=np.int32))
        assert len(result) == 0
        assert total == 0

    def test_single(self):
        result, total = prefix_sum_exclusive_multi(np.array([42], dtype=np.int32))
        np.testing.assert_array_equal(result, [0])
        assert total == 42

    def test_512_elements(self):
        rng = np.random.default_rng(0xCAFE)
        values = rng.integers(0, 100, size=512, dtype=np.int32)
        result, total = prefix_sum_exclusive_multi(values)
        expected = np.zeros(512, dtype=np.int32)
        acc = np.int32(0)
        for i in range(512):
            expected[i] = acc
            acc = np.int32(acc + values[i])
        np.testing.assert_array_equal(result, expected)
        assert total == int(acc)


class TestCPUMultiCompact:
    def test_basic(self):
        values = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        mask = np.array([1, 0, 1, 0, 1], dtype=np.int32)
        result, count = compact_multi(values, mask)
        np.testing.assert_array_equal(result, [10, 30, 50])
        assert count == 3

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="equal lengths"):
            compact_multi(np.array([1], dtype=np.int32), np.array([], dtype=np.int32))


class TestCPUMultiReduce:
    def test_basic(self):
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = reduce_sum_multi(values)
        assert result.value == pytest.approx(10.0)
        assert result.count == 4

    def test_empty(self):
        result = reduce_sum_multi(np.array([], dtype=np.float32))
        assert result.value == 0.0
        assert result.count == 0


class TestGPUMultiPrefixSum:
    @pytest.mark.parametrize("size", [256, 512, 1024, 4096])
    def test_cpu_parity(self, worker, size):
        rng = np.random.default_rng(0xBEEF + size)
        values = rng.integers(0, 50, size=size, dtype=np.int32)
        from chitin.gpu.dispatch_multi import dispatch_prefix_sum_exclusive_multi

        gpu_result, gpu_total = dispatch_prefix_sum_exclusive_multi(worker, values)
        cpu_result, cpu_total = prefix_sum_exclusive_multi(values)
        np.testing.assert_array_equal(gpu_result, cpu_result)
        assert gpu_total == cpu_total

    def test_non_power_of_two(self, worker):
        rng = np.random.default_rng(0xDEAD)
        values = rng.integers(0, 50, size=300, dtype=np.int32)
        from chitin.gpu.dispatch_multi import dispatch_prefix_sum_exclusive_multi

        gpu_result, gpu_total = dispatch_prefix_sum_exclusive_multi(worker, values)
        cpu_result, cpu_total = prefix_sum_exclusive_multi(values)
        np.testing.assert_array_equal(gpu_result, cpu_result)
        assert gpu_total == cpu_total

    def test_exactly_256(self, worker):
        rng = np.random.default_rng(0xFACE)
        values = rng.integers(0, 100, size=256, dtype=np.int32)
        from chitin.gpu.dispatch_multi import dispatch_prefix_sum_exclusive_multi

        gpu_result, gpu_total = dispatch_prefix_sum_exclusive_multi(worker, values)
        cpu_result, cpu_total = prefix_sum_exclusive_multi(values)
        np.testing.assert_array_equal(gpu_result, cpu_result)
        assert gpu_total == cpu_total

    def test_empty(self, worker):
        from chitin.gpu.dispatch_multi import dispatch_prefix_sum_exclusive_multi

        result, total = dispatch_prefix_sum_exclusive_multi(
            worker, np.array([], dtype=np.int32)
        )
        assert len(result) == 0
        assert total == 0


class TestGPUMultiCompact:
    @pytest.mark.parametrize("size", [512, 1024])
    def test_cpu_parity(self, worker, size):
        rng = np.random.default_rng(0xCAFE + size)
        values = rng.integers(0, 1000, size=size, dtype=np.int32)
        mask = rng.integers(0, 2, size=size, dtype=np.int32)
        from chitin.gpu.dispatch_multi import dispatch_compact_multi

        gpu_result, gpu_count = dispatch_compact_multi(worker, values, mask)
        cpu_result, cpu_count = compact_multi(values, mask)
        assert gpu_count == cpu_count
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_all_kept(self, worker):
        values = np.arange(300, dtype=np.int32)
        mask = np.ones(300, dtype=np.int32)
        from chitin.gpu.dispatch_multi import dispatch_compact_multi

        result, count = dispatch_compact_multi(worker, values, mask)
        assert count == 300
        np.testing.assert_array_equal(result, values)

    def test_none_kept(self, worker):
        values = np.arange(300, dtype=np.int32)
        mask = np.zeros(300, dtype=np.int32)
        from chitin.gpu.dispatch_multi import dispatch_compact_multi

        _, count = dispatch_compact_multi(worker, values, mask)
        assert count == 0


class TestGPUMultiReduce:
    @pytest.mark.parametrize("size", [512, 1024, 4096])
    def test_cpu_parity(self, worker, size):
        rng = np.random.default_rng(0xFEED + size)
        values = rng.random(size).astype(np.float32)
        from chitin.gpu.dispatch_multi import dispatch_reduce_sum_multi

        gpu_result = dispatch_reduce_sum_multi(worker, values)
        cpu_result = reduce_sum_multi(values)
        assert gpu_result == pytest.approx(cpu_result.value, rel=1e-4)

    def test_empty(self, worker):
        from chitin.gpu.dispatch_multi import dispatch_reduce_sum_multi

        result = dispatch_reduce_sum_multi(worker, np.array([], dtype=np.float32))
        assert result == 0.0
