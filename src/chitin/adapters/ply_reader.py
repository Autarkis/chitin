"""Minimal, permissively licensed PLY reader.

Reads the ``vertex`` element of a PLY file (ascii or binary, little- or
big-endian) into a NumPy structured array. Chitin only needs vertex attributes
(positions, opacity, gaussian-splat scale/rotation, normals), so any ``face`` or
other elements are parsed just enough to skip past them. This replaces the
GPL-licensed ``plyfile`` package so the whole dependency stack stays permissive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# PLY scalar type name -> NumPy dtype code (without endianness prefix).
_PLY_TO_NP = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


class PlyVertexElement:
    """Stand-in for a plyfile vertex element.

    Supports ``element["prop"]`` (returns a NumPy array), ``len(element)`` (the
    vertex count) and ``element.data`` (the underlying structured array, so
    callers can inspect ``element.data.dtype.names``).
    """

    __slots__ = ("data",)

    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    def __getitem__(self, key: str) -> np.ndarray:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)


def read_ply_vertex(path: str | Path) -> PlyVertexElement:
    """Read the ``vertex`` element of a PLY file into a structured array."""
    with open(path, "rb") as f:
        fmt, elements = _read_header(f, path)
        if fmt == "ascii":
            data = _read_ascii_vertex(f, elements)
        elif fmt == "binary_little_endian":
            data = _read_binary_vertex(f, elements, "<")
        elif fmt == "binary_big_endian":
            data = _read_binary_vertex(f, elements, ">")
        else:
            raise ValueError(f"unsupported PLY format: {fmt!r}")
    return PlyVertexElement(data)


def _read_header(f, path):
    if f.readline().strip() != b"ply":
        raise ValueError(f"not a PLY file: {path}")
    fmt = None
    elements: list[list] = []
    current = None
    while True:
        line = f.readline()
        if not line:
            raise ValueError("unexpected end of file in PLY header")
        tokens = line.split()
        if not tokens:
            continue
        keyword = tokens[0]
        if keyword == b"format":
            fmt = tokens[1].decode()
        elif keyword in (b"comment", b"obj_info"):
            continue
        elif keyword == b"element":
            current = [tokens[1].decode(), int(tokens[2]), []]
            elements.append(current)
        elif keyword == b"property":
            if current is None:
                raise ValueError("PLY property declared before any element")
            if tokens[1] == b"list":
                count_t = _PLY_TO_NP[tokens[2].decode()]
                item_t = _PLY_TO_NP[tokens[3].decode()]
                current[2].append(("list", count_t, item_t, tokens[4].decode()))
            else:
                current[2].append(
                    (tokens[2].decode(), _PLY_TO_NP[tokens[1].decode()])
                )
        elif keyword == b"end_header":
            break
    if fmt is None:
        raise ValueError("PLY header missing 'format' line")
    return fmt, elements


def _has_list(props) -> bool:
    return any(p[0] == "list" for p in props)


def _scalar_dtype(props, endian) -> np.dtype:
    return np.dtype([(name, endian + code) for (name, code) in props])


def _read_binary_vertex(f, elements, endian) -> np.ndarray:
    vertex = None
    for name, count, props in elements:
        if _has_list(props):
            if name == "vertex":
                raise ValueError("PLY 'vertex' with list properties is unsupported")
            _skip_binary_list_element(f, count, props, endian)
            continue
        dtype = _scalar_dtype(props, endian)
        nbytes = dtype.itemsize * count
        buf = f.read(nbytes)
        if len(buf) < nbytes:
            raise ValueError("unexpected end of file in PLY body")
        if name == "vertex":
            vertex = np.frombuffer(buf, dtype=dtype, count=count).copy()
    if vertex is None:
        raise ValueError("PLY has no 'vertex' element")
    return vertex


def _skip_binary_list_element(f, count, props, endian) -> None:
    for _ in range(count):
        for p in props:
            if p[0] == "list":
                _, count_t, item_t, _name = p
                ct = np.dtype(endian + count_t)
                raw = f.read(ct.itemsize)
                if len(raw) < ct.itemsize:
                    raise ValueError("unexpected end of file in PLY body")
                n = int(np.frombuffer(raw, dtype=ct, count=1)[0])
                f.read(np.dtype(endian + item_t).itemsize * n)
            else:
                f.read(np.dtype(endian + p[1]).itemsize)


def _read_ascii_vertex(f, elements) -> np.ndarray:
    vertex = None
    for name, count, props in elements:
        if name == "vertex":
            if _has_list(props):
                raise ValueError("PLY 'vertex' with list properties is unsupported")
            dtype = np.dtype([(pname, code) for (pname, code) in props])
            rows = [f.readline().decode().split() for _ in range(count)]
            if len(rows) < count or any(len(r) < len(props) for r in rows):
                raise ValueError("unexpected end of file in PLY body")
            arr = np.empty(count, dtype=dtype)
            for j, (pname, _code) in enumerate(props):
                arr[pname] = [r[j] for r in rows]
            vertex = arr
        else:
            for _ in range(count):
                if not f.readline():
                    raise ValueError("unexpected end of file in PLY body")
    if vertex is None:
        raise ValueError("PLY has no 'vertex' element")
    return vertex
