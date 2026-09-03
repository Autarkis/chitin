"""GPU buffer struct layouts and bounded arena allocator.

Struct layouts follow WebGPU std430-like alignment (vec3 padded to vec4,
scalars 4-byte aligned). Each layout is the authoritative source for
buffer sizing and field offsets — WGSL struct definitions are generated
from these, not duplicated.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum, auto

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

    def dtype(self) -> np.dtype:
        entries = []
        for f in self.fields.values():
            np_fmt = {"f": np.float32, "I": np.uint32, "i": np.int32}[f.fmt]
            shape = (f.count,) if f.count > 1 else ()
            entries.append((f.name, np_fmt, shape))
        return np.dtype(entries)

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


class ArenaKind(Enum):
    SINGLE_LARGE = auto()
    MULTIPART = auto()
    FRAGMENT_BATCH = auto()
    TILE = auto()


@dataclass
class ArenaDescriptor:
    kind: ArenaKind
    layout: StructLayout
    max_elements: int
    label: str = ""

    @property
    def byte_capacity(self) -> int:
        return self.layout.buffer_size(self.max_elements)


class BoundedArena:
    """Fixed-capacity GPU buffer arena. No unchecked growth.

    Tracks allocations as (offset, count) pairs. Raises CapacityError
    on overflow instead of silently growing.
    """

    def __init__(self, descriptor: ArenaDescriptor):
        self.descriptor = descriptor
        self._allocated = 0
        self._regions: list[tuple[int, int]] = []

    @property
    def capacity(self) -> int:
        return self.descriptor.max_elements

    @property
    def allocated(self) -> int:
        return self._allocated

    @property
    def remaining(self) -> int:
        return self.capacity - self._allocated

    def allocate(self, count: int) -> tuple[int, int]:
        if count <= 0:
            raise ValueError(f"Cannot allocate {count} elements")
        if self._allocated + count > self.capacity:
            from chitin.gpu.errors import CapacityError

            raise CapacityError(
                f"Arena {self.descriptor.label!r} overflow: "
                f"requested {count}, remaining {self.remaining}/{self.capacity}"
            )
        offset = self._allocated
        self._allocated += count
        self._regions.append((offset, count))
        return offset, count

    def reset(self) -> None:
        self._allocated = 0
        self._regions.clear()

    def byte_offset(self, element_offset: int) -> int:
        return element_offset * self.descriptor.layout.stride

    def __repr__(self) -> str:
        return (
            f"BoundedArena({self.descriptor.label!r}, "
            f"{self._allocated}/{self.capacity})"
        )


class ArenaSet:
    """Named collection of arenas for a compute pass."""

    def __init__(self) -> None:
        self._arenas: dict[str, BoundedArena] = {}

    def add(self, name: str, descriptor: ArenaDescriptor) -> BoundedArena:
        if name in self._arenas:
            raise ValueError(f"Arena {name!r} already exists")
        arena = BoundedArena(descriptor)
        self._arenas[name] = arena
        return arena

    def get(self, name: str) -> BoundedArena:
        return self._arenas[name]

    def reset_all(self) -> None:
        for arena in self._arenas.values():
            arena.reset()

    def summary(self) -> dict[str, dict[str, int]]:
        return {
            name: {
                "allocated": a.allocated,
                "capacity": a.capacity,
                "bytes": a.byte_offset(a.allocated),
            }
            for name, a in self._arenas.items()
        }

    def __contains__(self, name: str) -> bool:
        return name in self._arenas

    def __len__(self) -> int:
        return len(self._arenas)
