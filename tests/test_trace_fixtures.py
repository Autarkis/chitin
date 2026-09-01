"""Fixture mesh sanity checks."""

import numpy as np
import pytest

from chitin.trace_fixtures import FIXTURES


@pytest.mark.parametrize("name,factory", list(FIXTURES.items()))
class TestFixtureMesh:
    def test_returns_vertices_and_faces(self, name, factory):
        v, f = factory()
        assert isinstance(v, np.ndarray)
        assert isinstance(f, np.ndarray)
        assert v.ndim == 2 and v.shape[1] == 3
        assert f.ndim == 2 and f.shape[1] == 3

    def test_vertex_dtype_float32(self, name, factory):
        v, _ = factory()
        assert v.dtype == np.float32

    def test_face_dtype_int32(self, name, factory):
        _, f = factory()
        assert f.dtype == np.int32

    def test_face_indices_valid(self, name, factory):
        v, f = factory()
        assert f.min() >= 0
        assert f.max() < len(v)

    def test_has_faces(self, name, factory):
        _, f = factory()
        assert len(f) >= 1

    def test_deterministic(self, name, factory):
        v1, f1 = factory()
        v2, f2 = factory()
        np.testing.assert_array_equal(v1, v2)
        np.testing.assert_array_equal(f1, f2)
