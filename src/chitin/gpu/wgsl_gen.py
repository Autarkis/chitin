"""Generate WGSL struct definitions from authoritative Python layouts.

The Python StructLayout definitions in layouts.py are the single source
of truth. WGSL structs are generated, not hand-written, to prevent
layout drift between CPU and GPU code.
"""

from __future__ import annotations

from chitin.gpu.layouts import ALL_LAYOUTS, StructLayout

_FMT_TO_WGSL: dict[str, str] = {
    "f": "f32",
    "I": "u32",
    "i": "i32",
}


def layout_to_wgsl(layout: StructLayout) -> str:
    """Generate a WGSL struct definition from a StructLayout."""
    lines = [f"struct {layout.name} {{"]
    for field in layout.fields.values():
        wgsl_type = _FMT_TO_WGSL.get(field.fmt)
        if wgsl_type is None:
            raise ValueError(f"No WGSL mapping for format {field.fmt!r}")
        if field.count > 1:
            wgsl_type = f"array<{wgsl_type}, {field.count}>"
        lines.append(f"    {field.name}: {wgsl_type},")
    lines.append("}")
    return "\n".join(lines)


def generate_all_structs() -> str:
    """Generate WGSL definitions for all registered layouts."""
    blocks = []
    for name in sorted(ALL_LAYOUTS.keys()):
        blocks.append(layout_to_wgsl(ALL_LAYOUTS[name]))
    return "\n\n".join(blocks)


def generate_struct_by_name(name: str) -> str:
    """Generate WGSL for a single named layout."""
    if name not in ALL_LAYOUTS:
        raise KeyError(f"Unknown layout: {name!r}")
    return layout_to_wgsl(ALL_LAYOUTS[name])


def check_drift(wgsl_source: str) -> list[str]:
    """Compare existing WGSL struct source against generated definitions.

    Returns a list of drift messages. Empty list means no drift.
    """
    diffs = []
    for name, layout in ALL_LAYOUTS.items():
        expected = layout_to_wgsl(layout)
        struct_header = f"struct {name} {{"
        if struct_header not in wgsl_source:
            diffs.append(f"missing struct {name}")
            continue
        start = wgsl_source.index(struct_header)
        end = wgsl_source.index("}", start) + 1
        actual = wgsl_source[start:end]
        if actual.strip() != expected.strip():
            diffs.append(f"struct {name} has drifted from authoritative layout")
    return diffs
