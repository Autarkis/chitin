"""Tests for GPU buffer struct layouts and bounded arena allocator."""

import numpy as np
import pytest

from chitin.gpu.errors import CapacityError
from chitin.gpu.layouts import (
    ALL_LAYOUTS,
    HULL_HEADER,
    INTERSECTION_POINT,
    MESH_HEADER,
    OUTPUT_HEADER,
    PLANE,
    POINT,
    TRIANGLE,
    ArenaDescriptor,
    ArenaKind,
    ArenaSet,
    BoundedArena,
)


class TestStructLayouts:
    def test_point_stride_16(self):
        assert POINT.stride == 16

    def test_triangle_stride_16(self):
        assert TRIANGLE.stride == 16

    def test_plane_stride_16(self):
        assert PLANE.stride == 16

    def test_mesh_header_stride_16(self):
        assert MESH_HEADER.stride == 16

    def test_intersection_point_has_source_edge(self):
        assert "source_edge" in INTERSECTION_POINT.fields
        assert INTERSECTION_POINT.fields["source_edge"].fmt == "I"

    def test_hull_header_stride_32(self):
        assert HULL_HEADER.stride == 32

    def test_output_header_stride_16(self):
        assert OUTPUT_HEADER.stride == 16

    def test_buffer_size(self):
        assert POINT.buffer_size(100) == 1600
        assert HULL_HEADER.buffer_size(10) == 320

    def test_dtype_produces_structured_array(self):
        dt = POINT.dtype()
        arr = np.zeros(3, dtype=dt)
        arr[0]["x"] = 1.5
        assert arr[0]["x"] == pytest.approx(1.5)

    def test_all_layouts_registered(self):
        assert len(ALL_LAYOUTS) == 7
        assert "Point" in ALL_LAYOUTS
        assert "HullHeader" in ALL_LAYOUTS

    @pytest.mark.parametrize("name", ALL_LAYOUTS.keys())
    def test_layout_stride_multiple_of_16(self, name):
        assert ALL_LAYOUTS[name].stride % 16 == 0

    @pytest.mark.parametrize("name", ALL_LAYOUTS.keys())
    def test_fields_within_stride(self, name):
        layout = ALL_LAYOUTS[name]
        for f in layout.fields.values():
            assert f.offset + f.size <= layout.stride


class TestBoundedArena:
    def _make_arena(self, max_elements=100):
        desc = ArenaDescriptor(
            kind=ArenaKind.SINGLE_LARGE,
            layout=POINT,
            max_elements=max_elements,
            label="test",
        )
        return BoundedArena(desc)

    def test_allocate_returns_offset_count(self):
        arena = self._make_arena()
        offset, count = arena.allocate(10)
        assert offset == 0
        assert count == 10

    def test_sequential_allocation(self):
        arena = self._make_arena()
        o1, _ = arena.allocate(10)
        o2, _ = arena.allocate(20)
        assert o1 == 0
        assert o2 == 10
        assert arena.allocated == 30
        assert arena.remaining == 70

    def test_overflow_raises_capacity_error(self):
        arena = self._make_arena(10)
        arena.allocate(8)
        with pytest.raises(CapacityError):
            arena.allocate(5)

    def test_zero_allocation_raises(self):
        arena = self._make_arena()
        with pytest.raises(ValueError):
            arena.allocate(0)

    def test_reset_clears(self):
        arena = self._make_arena()
        arena.allocate(50)
        arena.reset()
        assert arena.allocated == 0
        assert arena.remaining == 100

    def test_byte_offset(self):
        arena = self._make_arena()
        assert arena.byte_offset(5) == 5 * 16

    def test_byte_capacity(self):
        arena = self._make_arena(100)
        assert arena.descriptor.byte_capacity == 1600


class TestArenaSet:
    def test_add_and_get(self):
        aset = ArenaSet()
        desc = ArenaDescriptor(
            kind=ArenaKind.FRAGMENT_BATCH,
            layout=TRIANGLE,
            max_elements=1000,
            label="tris",
        )
        arena = aset.add("triangles", desc)
        assert aset.get("triangles") is arena
        assert "triangles" in aset
        assert len(aset) == 1

    def test_duplicate_name_raises(self):
        aset = ArenaSet()
        desc = ArenaDescriptor(
            kind=ArenaKind.TILE,
            layout=POINT,
            max_elements=100,
            label="pts",
        )
        aset.add("points", desc)
        with pytest.raises(ValueError):
            aset.add("points", desc)

    def test_reset_all(self):
        aset = ArenaSet()
        for name, layout in [("a", POINT), ("b", TRIANGLE)]:
            desc = ArenaDescriptor(
                kind=ArenaKind.MULTIPART,
                layout=layout,
                max_elements=50,
                label=name,
            )
            arena = aset.add(name, desc)
            arena.allocate(25)
        aset.reset_all()
        assert aset.get("a").allocated == 0
        assert aset.get("b").allocated == 0

    def test_summary(self):
        aset = ArenaSet()
        desc = ArenaDescriptor(
            kind=ArenaKind.SINGLE_LARGE,
            layout=POINT,
            max_elements=100,
            label="pts",
        )
        arena = aset.add("pts", desc)
        arena.allocate(10)
        s = aset.summary()
        assert s["pts"]["allocated"] == 10
        assert s["pts"]["capacity"] == 100
        assert s["pts"]["bytes"] == 160
