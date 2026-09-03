"""GPU buffer struct layouts.

Each layout is the authoritative source for NumPy packing, buffer sizing, and
WGSL field order. The currently supported scalar fields deliberately require
their declared offsets to match WGSL's natural layout; a future vector/matrix
extension must add its alignment rule here before it can be used in a shader.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FieldDef:
    name: str
    fmt: str
    offset: int
    count: int = 1

    @property
    def size(self) -> int:
        return struct.calcsize(self.fmt) * self.count


class StructLayout:
    """Describes a GPU struct layout with explicit field offsets and alignment."""

    def __init__(self, name: str, fields: list[FieldDef], stride: int):
        self.name = name
        self.fields = {f.name: f for f in fields}
        self.stride = stride
        self._validate(fields)

    @staticmethod
    def _field_dtype(field: FieldDef) -> np.dtype:
        try:
            return np.dtype({"f": np.float32, "I": np.uint32, "i": np.int32}[field.fmt])
        except KeyError as exc:
            raise ValueError(f"Unsupported GPU field format {field.fmt!r}") from exc

    def _validate(self, fields: list[FieldDef]) -> None:
        if self.stride <= 0 or self.stride % 4:
            raise ValueError("GPU struct stride must be a positive multiple of 4")
        if len(self.fields) != len(fields):
            raise ValueError(f"Struct {self.name!r} has duplicate field names")

        cursor = 0
        for field in fields:
            dtype = self._field_dtype(field)
            if field.count <= 0:
                raise ValueError(f"Field {field.name!r} must have a positive count")
            if field.offset != cursor:
                raise ValueError(
                    f"Field {field.name!r} offset {field.offset} does not match "
                    f"WGSL scalar layout offset {cursor}; add an explicit padding field"
                )
            cursor += dtype.itemsize * field.count

        if cursor > self.stride:
            raise ValueError(
                f"Struct {self.name!r} fields require {cursor} bytes, exceeding "
                f"stride {self.stride}"
            )

    def dtype(self) -> np.dtype:
        """Return a dtype whose offsets and itemsize exactly match this layout."""
        fields = list(self.fields.values())
        formats = [
            (self._field_dtype(field), (field.count,))
            if field.count > 1
            else self._field_dtype(field)
            for field in fields
        ]
        return np.dtype(
            {
                "names": [field.name for field in fields],
                "formats": formats,
                "offsets": [field.offset for field in fields],
                "itemsize": self.stride,
            }
        )

    def buffer_size(self, count: int) -> int:
        return self.stride * count

    def __repr__(self) -> str:
        return f"StructLayout({self.name!r}, stride={self.stride})"


POINT = StructLayout(
    "Point",
    [
        FieldDef("x", "f", 0),
        FieldDef("y", "f", 4),
        FieldDef("z", "f", 8),
        FieldDef("_pad0", "f", 12),
    ],
    stride=16,
)

TRIANGLE = StructLayout(
    "Triangle",
    [
        FieldDef("i0", "I", 0),
        FieldDef("i1", "I", 4),
        FieldDef("i2", "I", 8),
        FieldDef("_pad0", "I", 12),
    ],
    stride=16,
)

PLANE = StructLayout(
    "Plane",
    [
        FieldDef("a", "f", 0),
        FieldDef("b", "f", 4),
        FieldDef("c", "f", 8),
        FieldDef("d", "f", 12),
    ],
    stride=16,
)

MESH_HEADER = StructLayout(
    "MeshHeader",
    [
        FieldDef("vertex_offset", "I", 0),
        FieldDef("vertex_count", "I", 4),
        FieldDef("face_offset", "I", 8),
        FieldDef("face_count", "I", 12),
    ],
    stride=16,
)

INTERSECTION_POINT = StructLayout(
    "IntersectionPoint",
    [
        FieldDef("x", "f", 0),
        FieldDef("y", "f", 4),
        FieldDef("z", "f", 8),
        FieldDef("source_edge", "I", 12),
    ],
    stride=16,
)

HULL_HEADER = StructLayout(
    "HullHeader",
    [
        FieldDef("vertex_offset", "I", 0),
        FieldDef("vertex_count", "I", 4),
        FieldDef("face_offset", "I", 8),
        FieldDef("face_count", "I", 12),
        FieldDef("volume", "f", 16),
        FieldDef("status", "I", 20),
        FieldDef("_pad0", "I", 24),
        FieldDef("_pad1", "I", 28),
    ],
    stride=32,
)

OUTPUT_HEADER = StructLayout(
    "OutputHeader",
    [
        FieldDef("hull_count", "I", 0),
        FieldDef("total_vertices", "I", 4),
        FieldDef("total_faces", "I", 8),
        FieldDef("error_code", "I", 12),
    ],
    stride=16,
)

ALL_LAYOUTS: dict[str, StructLayout] = {
    "Point": POINT,
    "Triangle": TRIANGLE,
    "Plane": PLANE,
    "MeshHeader": MESH_HEADER,
    "IntersectionPoint": INTERSECTION_POINT,
    "HullHeader": HULL_HEADER,
    "OutputHeader": OUTPUT_HEADER,
}
