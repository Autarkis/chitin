"""Tests for WGSL generation from authoritative layouts."""

import pytest

from chitin.gpu.layouts import ALL_LAYOUTS, HULL_HEADER, PLANE, POINT
from chitin.gpu.wgsl_gen import (
    check_drift,
    generate_all_structs,
    generate_struct_by_name,
    layout_to_wgsl,
)


class TestLayoutToWGSL:
    def test_point_struct(self):
        wgsl = layout_to_wgsl(POINT)
        assert "struct Point {" in wgsl
        assert "x: f32," in wgsl
        assert "y: f32," in wgsl
        assert "z: f32," in wgsl
        assert wgsl.endswith("}")

    def test_plane_struct(self):
        wgsl = layout_to_wgsl(PLANE)
        assert "struct Plane {" in wgsl
        assert "a: f32," in wgsl
        assert "d: f32," in wgsl

    def test_hull_header_has_volume_and_status(self):
        wgsl = layout_to_wgsl(HULL_HEADER)
        assert "volume: f32," in wgsl
        assert "status: u32," in wgsl

    def test_all_layouts_generate(self):
        for name, layout in ALL_LAYOUTS.items():
            wgsl = layout_to_wgsl(layout)
            assert f"struct {name} {{" in wgsl

    def test_unknown_format_raises(self):
        from chitin.gpu.layouts import FieldDef, StructLayout

        bad = StructLayout("Bad", [FieldDef("x", "Q", 0)], stride=8)
        with pytest.raises(ValueError, match="No WGSL mapping"):
            layout_to_wgsl(bad)


class TestGenerateAll:
    def test_contains_all_structs(self):
        wgsl = generate_all_structs()
        for name in ALL_LAYOUTS:
            assert f"struct {name} {{" in wgsl

    def test_by_name(self):
        wgsl = generate_struct_by_name("Point")
        assert "struct Point {" in wgsl

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            generate_struct_by_name("NonExistent")


class TestDriftCheck:
    def test_no_drift_on_generated(self):
        wgsl = generate_all_structs()
        diffs = check_drift(wgsl)
        assert diffs == []

    def test_missing_struct_detected(self):
        wgsl = generate_all_structs()
        wgsl = wgsl.replace("struct Point {", "struct Pointy {")
        diffs = check_drift(wgsl)
        assert any("missing struct Point" in d for d in diffs)

    def test_drifted_struct_detected(self):
        wgsl = generate_all_structs()
        wgsl = wgsl.replace("x: f32,", "x: u32,")
        diffs = check_drift(wgsl)
        assert len(diffs) > 0
        assert any("drifted" in d for d in diffs)
