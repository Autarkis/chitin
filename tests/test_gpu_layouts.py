"""Tests for authoritative GPU buffer struct layouts."""

import numpy as np
import pytest

from chitin.gpu.layouts import (
    ALL_LAYOUTS,
    HULL_HEADER,
    INTERSECTION_POINT,
    MESH_HEADER,
    OUTPUT_HEADER,
    PLANE,
    POINT,
    TRIANGLE,
    FieldDef,
    StructLayout,
)


class TestStructLayouts:
    @pytest.mark.parametrize(
        ("layout", "stride"),
        [
            (POINT, 16),
            (TRIANGLE, 16),
            (PLANE, 16),
            (MESH_HEADER, 16),
            (INTERSECTION_POINT, 16),
            (HULL_HEADER, 32),
            (OUTPUT_HEADER, 16),
        ],
    )
    def test_stride(self, layout, stride):
        assert layout.stride == stride

    def test_dtype_preserves_declared_offsets_and_stride(self):
        dtype = HULL_HEADER.dtype()
        assert dtype.itemsize == HULL_HEADER.stride
        for name, field in HULL_HEADER.fields.items():
            assert dtype.fields[name][1] == field.offset

    def test_dtype_produces_uploadable_structured_array(self):
        array = np.zeros(3, dtype=POINT.dtype())
        array[0]["x"] = 1.5
        assert array.nbytes == POINT.buffer_size(3)
        assert array[0]["x"] == pytest.approx(1.5)

    def test_all_layouts_registered(self):
        assert set(ALL_LAYOUTS) == {
            "Point",
            "Triangle",
            "Plane",
            "MeshHeader",
            "IntersectionPoint",
            "HullHeader",
            "OutputHeader",
        }

    @pytest.mark.parametrize("layout", ALL_LAYOUTS.values())
    def test_fields_fit_exactly_within_stride(self, layout):
        for field in layout.fields.values():
            assert field.offset + field.size <= layout.stride

    def test_rejects_implicit_layout_gap(self):
        with pytest.raises(ValueError, match="explicit padding"):
            StructLayout("Bad", [FieldDef("x", "f", 4)], stride=8)

    def test_rejects_duplicate_field_names(self):
        with pytest.raises(ValueError, match="duplicate"):
            StructLayout(
                "Bad",
                [FieldDef("x", "f", 0), FieldDef("x", "f", 4)],
                stride=8,
            )
