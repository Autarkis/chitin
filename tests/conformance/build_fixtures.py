"""Author the cross-runtime .phys conformance corpus + its manifest.

Run once to (re)generate the frozen golden binaries + ``manifest.json`` that the
Python (`test_conformance.py`) and web (`integrations/web/test/conformance.test.ts`)
readers both verify against:

    PYTHONPATH=src python tests/conformance/build_fixtures.py

Fixtures are authored with the **production writer** (`chitin.exporters.phys.export_phys`)
so the corpus can't drift from the real encoder; the manifest is then derived by
reading each valid fixture back with the **production reader** (`read_phys`), so the
two readers are checked against the same independently-recorded expectations. Invalid
fixtures are produced by corrupting a valid one and carry an expected error substring.

The manifest only records LOD-tier *internals* for the Python reader; the web parser
intentionally skips LOD tier bodies (exposing `hasLod` + the LOD0 hulls only), so the
cross-runtime field set is version/flags/has* / hull count / per-hull AABB+counts /
totals / bones / validity.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from chitin.exporters.phys import export_phys
from chitin.phys import LOD_TIER_HEADER_SIZE, LodTier, PhysBone, PhysHull, read_phys

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"

# Unit cube: 8 corners, 12 triangles (36 indices).
_CUBE_V = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.float32,
)
_CUBE_I = np.array(
    [
        0,
        1,
        2,
        0,
        2,
        3,  # bottom
        4,
        6,
        5,
        4,
        7,
        6,  # top
        0,
        4,
        5,
        0,
        5,
        1,  # front
        1,
        5,
        6,
        1,
        6,
        2,  # right
        2,
        6,
        7,
        2,
        7,
        3,  # back
        3,
        7,
        4,
        3,
        4,
        0,  # left
    ],
    dtype=np.uint16,
)
# A coarser hull (tetrahedron) for a low-detail LOD tier.
_TETRA_V = np.array([[0, 0, 0], [2, 0, 0], [1, 2, 0], [1, 1, 2]], dtype=np.float32)
_TETRA_I = np.array([0, 1, 2, 0, 1, 3, 1, 2, 3, 0, 2, 3], dtype=np.uint16)


def _hull(
    verts: np.ndarray, idx: np.ndarray, bone_index: int | None = None
) -> PhysHull:
    return PhysHull(
        vertices=verts.copy(),
        indices=idx.copy(),
        aabb_min=verts.min(axis=0),
        aabb_max=verts.max(axis=0),
        bone_index=bone_index,
    )


def _result(hulls, bones=None, lod_tiers=None) -> Any:
    # Duck-types chitin.result.ExtractionResult for export_phys (.hulls/.bones/.lod_tiers).
    return SimpleNamespace(hulls=hulls, bones=bones, lod_tiers=lod_tiers)


def _write_valid_fixtures() -> dict[str, str]:
    """Author the valid fixtures; return {filename: short description}."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    specs: dict[str, str] = {}

    # static: one convex hull, no bones / LOD.
    export_phys(_result([_hull(_CUBE_V, _CUBE_I)]), FIXTURES / "static_hull.phys")
    specs["static_hull.phys"] = "single convex hull, no bones/LOD"

    # multi-LOD: LOD0 cube + two ascending-concavity tiers.
    export_phys(
        _result(
            [_hull(_CUBE_V, _CUBE_I)],
            lod_tiers=[
                LodTier(concavity=0.01, hulls=[_hull(_CUBE_V, _CUBE_I)]),
                LodTier(concavity=0.05, hulls=[_hull(_TETRA_V, _TETRA_I)]),
            ],
        ),
        FIXTURES / "multi_lod.phys",
    )
    specs["multi_lod.phys"] = "LOD0 + two LOD tiers (ascending concavity)"

    # rigged: two hulls each bound to a bone, with bind poses.
    bind0 = np.eye(4, dtype=np.float32)
    bind1 = np.eye(4, dtype=np.float32)
    bind1[3, :3] = [1.0, 0.0, 0.0]  # translation in last row (row-vector convention)
    export_phys(
        _result(
            [
                _hull(_CUBE_V, _CUBE_I, bone_index=0),
                _hull(_TETRA_V, _TETRA_I, bone_index=1),
            ],
            bones=[
                PhysBone(name="root", bind_transform=bind0),
                PhysBone(name="child", bind_transform=bind1),
            ],
        ),
        FIXTURES / "rigged.phys",
    )
    specs["rigged.phys"] = "two hulls + two bones with bind poses"
    return specs


def _manifest_entry(name: str, description: str) -> dict:
    pf = read_phys(FIXTURES / name)
    return {
        "description": description,
        "valid": True,
        "version": pf.version,
        "flags": pf.flags,
        "hasBones": pf.has_bones,
        "hasBindPoses": pf.has_bind_poses,
        "hasLod": pf.has_lod,
        "hullCount": len(pf.hulls),
        "totalVertices": pf.total_vertices,
        "totalIndices": pf.total_indices,
        "totalTriangles": pf.total_triangles,
        "hulls": [
            {
                "vertexCount": int(len(h.vertices)),
                "indexCount": int(len(h.indices)),
                "aabbMin": [round(float(x), 6) for x in h.aabb_min],
                "aabbMax": [round(float(x), 6) for x in h.aabb_max],
                "boneIndex": h.bone_index,
            }
            for h in pf.hulls
        ],
        "bones": [{"name": b.name} for b in pf.bones],
        # LOD-tier internals are Python-only (the web parser skips tier bodies).
        "lodTiers": [
            {"concavity": round(float(t.concavity), 6), "hullCount": t.hull_count}
            for t in pf.lod_tiers
        ],
    }


def _write_invalid_fixtures() -> dict[str, dict]:
    """Corrupt a valid fixture in distinct ways; return manifest entries."""
    base = (FIXTURES / "static_hull.phys").read_bytes()
    entries: dict[str, dict] = {}

    bad_magic = b"XXXX" + base[4:]
    (FIXTURES / "invalid_magic.phys").write_bytes(bad_magic)
    entries["invalid_magic.phys"] = {
        "description": "wrong magic bytes",
        "valid": False,
        "errorContains": "magic",
    }

    bad_version = bytearray(base)
    struct.pack_into("<H", bad_version, 4, 99)  # version field
    (FIXTURES / "invalid_version.phys").write_bytes(bytes(bad_version))
    entries["invalid_version.phys"] = {
        "description": "unsupported version 99",
        "valid": False,
        "errorContains": "version",
    }

    (FIXTURES / "invalid_truncated.phys").write_bytes(base[:-8])
    entries["invalid_truncated.phys"] = {
        "description": "data truncated (last 8 bytes removed)",
        "valid": False,
        "errorContains": "truncat",
    }

    (FIXTURES / "invalid_trailing.phys").write_bytes(base + b"\x00\x00\x00\x00")
    entries["invalid_trailing.phys"] = {
        "description": "trailing bytes after end of data",
        "valid": False,
        "errorContains": "trailing",
    }

    # hull 0 vertex_offset pushed past the end so its span exceeds total_vertices.
    bad_voff = bytearray(base)
    total_verts = struct.unpack_from("<I", bad_voff, 12)[0]
    htoff = struct.unpack_from("<I", bad_voff, 20)[0]
    struct.pack_into("<I", bad_voff, htoff, total_verts)
    (FIXTURES / "invalid_vertex_offset.phys").write_bytes(bytes(bad_voff))
    entries["invalid_vertex_offset.phys"] = {
        "description": "hull vertex range exceeds total_vertices",
        "valid": False,
        "errorContains": "vertex range",
    }

    # hull 0 aabb_min.x set to NaN (finiteness check; NaN slips past min>max).
    bad_nan = bytearray(base)
    b_htoff = struct.unpack_from("<I", bad_nan, 20)[0]
    struct.pack_into("<f", bad_nan, b_htoff + 16, float("nan"))
    (FIXTURES / "invalid_nan_aabb.phys").write_bytes(bytes(bad_nan))
    entries["invalid_nan_aabb.phys"] = {
        "description": "hull aabb has a non-finite (NaN) component",
        "valid": False,
        "errorContains": "non-finite",
    }

    # rigged hull 0 bone_index set out of range (>= bone_count).
    rigged = (FIXTURES / "rigged.phys").read_bytes()
    bad_bone = bytearray(rigged)
    r_htoff = struct.unpack_from("<I", bad_bone, 20)[0]
    struct.pack_into("<i", bad_bone, r_htoff + 40, 999)
    (FIXTURES / "invalid_bone_index.phys").write_bytes(bytes(bad_bone))
    entries["invalid_bone_index.phys"] = {
        "description": "hull bone_index out of range",
        "valid": False,
        "errorContains": "bone_index",
    }

    # rigged hull 1 vertex_offset reset to 0 so its range overlaps hull 0
    # (still within total_vertices, so only the contiguity check catches it).
    # rigged has bones, so the descriptor size is 44 bytes.
    bad_overlap = bytearray(rigged)
    struct.pack_into("<I", bad_overlap, r_htoff + 44, 0)
    (FIXTURES / "invalid_overlapping_range.phys").write_bytes(bytes(bad_overlap))
    entries["invalid_overlapping_range.phys"] = {
        "description": "hull vertex range overlaps a previous hull",
        "valid": False,
        "errorContains": "contiguous",
    }

    # rigged bone bind transform corrupted to NaN (finite check on the bind block).
    (_v, _fl, _hc, _tv, r_tidx, _hto, _vdo, r_idata) = struct.unpack_from(
        "<HHIIIIII", rigged, 4
    )
    r_bone_block = r_idata + r_tidx * 2  # rigged has no LOD block
    bad_bind = bytearray(rigged)
    struct.pack_into("<f", bad_bind, r_bone_block + 4, float("nan"))  # skip bone_count
    (FIXTURES / "invalid_nan_bind.phys").write_bytes(bytes(bad_bind))
    entries["invalid_nan_bind.phys"] = {
        "description": "bone bind_transform has a non-finite component",
        "valid": False,
        "errorContains": "bind_transform",
    }

    # LOD-internal corruption. The web parser skips LOD tier bodies, so these are
    # Python-only (the web conformance test skips pythonOnly fixtures).
    lod = (FIXTURES / "multi_lod.phys").read_bytes()
    (_v2, _fl2, _hc2, _tv2, l_tidx, _hto2, _vdo2, l_idata) = struct.unpack_from(
        "<HHIIIIII", lod, 4
    )
    # multi_lod has no bind poses, so the LOD block follows the index data.
    l_tier0_hull0 = l_idata + l_tidx * 2 + 4 + LOD_TIER_HEADER_SIZE

    bad_lod_nan = bytearray(lod)
    struct.pack_into("<f", bad_lod_nan, l_tier0_hull0 + 16, float("nan"))
    (FIXTURES / "invalid_lod_nan_aabb.phys").write_bytes(bytes(bad_lod_nan))
    entries["invalid_lod_nan_aabb.phys"] = {
        "description": "LOD tier hull aabb has a non-finite component",
        "valid": False,
        "errorContains": "non-finite",
        "pythonOnly": True,
    }

    bad_lod_off = bytearray(lod)
    struct.pack_into("<I", bad_lod_off, l_tier0_hull0, 1)  # vertex_offset out of range
    (FIXTURES / "invalid_lod_offset.phys").write_bytes(bytes(bad_lod_off))
    entries["invalid_lod_offset.phys"] = {
        "description": "LOD tier hull vertex range leaves the tier",
        "valid": False,
        "errorContains": "range",
        "pythonOnly": True,
    }
    return entries


def main() -> None:
    valid_specs = _write_valid_fixtures()
    manifest: dict[str, dict] = {
        name: _manifest_entry(name, desc) for name, desc in valid_specs.items()
    }
    manifest.update(_write_invalid_fixtures())

    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(manifest)} fixtures + manifest.json to {HERE}")


if __name__ == "__main__":
    main()
