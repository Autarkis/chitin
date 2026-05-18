# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"PHYS"
CURRENT_VERSION = 2
HEADER_SIZE = 32
FLAG_HAS_BONES = 0x01
FLAG_HAS_BIND_POSES = 0x02


@dataclass
class PhysHull:
    vertices: np.ndarray
    indices: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    bone_index: int | None = None


@dataclass
class PhysBone:
    name: str
    bind_transform: np.ndarray


@dataclass
class PhysFile:
    version: int
    flags: int
    hulls: list[PhysHull]
    bones: list[PhysBone] = field(default_factory=list)

    @property
    def has_bones(self) -> bool:
        return bool(self.flags & FLAG_HAS_BONES)

    @property
    def has_bind_poses(self) -> bool:
        return bool(self.flags & FLAG_HAS_BIND_POSES)

    @property
    def total_vertices(self) -> int:
        return sum(len(h.vertices) for h in self.hulls)

    @property
    def total_indices(self) -> int:
        return sum(len(h.indices) for h in self.hulls)

    @property
    def total_triangles(self) -> int:
        return self.total_indices // 3


def _descriptor_size(flags: int) -> int:
    return 44 if flags & FLAG_HAS_BONES else 40


def read_phys(path: str | Path) -> PhysFile:
    data = Path(path).read_bytes()
    if len(data) < HEADER_SIZE:
        raise ValueError(f"file too small: {len(data)} bytes")

    if data[:4] != MAGIC:
        raise ValueError(f"bad magic: {data[:4]!r}")

    (
        version,
        flags,
        hull_count,
        total_verts,
        total_idx,
        hull_table_off,
        vertex_data_off,
        index_data_off,
    ) = struct.unpack_from("<HHIIIIII", data, 4)

    if version > CURRENT_VERSION:
        raise ValueError(f"unsupported version: {version}")

    has_bones = bool(flags & FLAG_HAS_BONES)
    desc_size = _descriptor_size(flags)

    hulls: list[PhysHull] = []
    off = hull_table_off
    for _ in range(hull_count):
        v_off, v_count, i_off, i_count = struct.unpack_from("<IIII", data, off)
        aabb_min = np.array(struct.unpack_from("<3f", data, off + 16), dtype=np.float32)
        aabb_max = np.array(struct.unpack_from("<3f", data, off + 28), dtype=np.float32)
        bone_idx = None
        if has_bones:
            raw_bone = struct.unpack_from("<i", data, off + 40)[0]
            bone_idx = None if raw_bone == -1 else raw_bone
        off += desc_size

        v_byte_off = vertex_data_off + v_off * 6
        raw = np.frombuffer(data, dtype=np.int16, count=v_count * 3, offset=v_byte_off)
        quantized = raw.reshape(-1, 3).astype(np.float32)
        extent = aabb_max - aabb_min
        extent = np.where(extent == 0, 1.0, extent)
        vertices = (quantized + 32768) / 65535 * extent + aabb_min

        i_byte_off = index_data_off + i_off * 2
        indices = np.frombuffer(
            data, dtype=np.uint16, count=i_count, offset=i_byte_off
        ).copy()

        hulls.append(
            PhysHull(
                vertices=vertices,
                indices=indices,
                aabb_min=aabb_min,
                aabb_max=aabb_max,
                bone_index=bone_idx,
            )
        )

    bones: list[PhysBone] = []
    if flags & FLAG_HAS_BIND_POSES:
        bone_off = index_data_off + total_idx * 2
        bone_count = struct.unpack_from("<I", data, bone_off)[0]
        bone_off += 4
        for _ in range(bone_count):
            mat = (
                np.frombuffer(data, dtype=np.float32, count=16, offset=bone_off)
                .reshape(4, 4)
                .copy()
            )
            bone_off += 64
            name_len = struct.unpack_from("<H", data, bone_off)[0]
            bone_off += 2
            name = data[bone_off : bone_off + name_len].decode("utf-8")
            bone_off += name_len
            bones.append(PhysBone(name=name, bind_transform=mat))

    return PhysFile(version=version, flags=flags, hulls=hulls, bones=bones)


@dataclass
class ValidationIssue:
    severity: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.message}"


def validate_phys(path: str | Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    data = Path(path).read_bytes()

    if len(data) < HEADER_SIZE:
        return [ValidationIssue("error", f"file too small: {len(data)} bytes")]

    if data[:4] != MAGIC:
        return [ValidationIssue("error", f"bad magic: {data[:4]!r}")]

    (
        version,
        flags,
        hull_count,
        total_verts,
        total_idx,
        hull_table_off,
        vertex_data_off,
        index_data_off,
    ) = struct.unpack_from("<HHIIIIII", data, 4)

    if version > CURRENT_VERSION:
        issues.append(ValidationIssue("error", f"unsupported version: {version}"))
        return issues

    has_bones = bool(flags & FLAG_HAS_BONES)
    desc_size = _descriptor_size(flags)

    if hull_table_off != HEADER_SIZE:
        issues.append(
            ValidationIssue(
                "error",
                f"hull_table_offset {hull_table_off} != expected {HEADER_SIZE}",
            )
        )

    expected_vertex_off = hull_table_off + hull_count * desc_size
    if vertex_data_off != expected_vertex_off:
        issues.append(
            ValidationIssue(
                "error",
                f"vertex_data_offset {vertex_data_off} != expected {expected_vertex_off}",
            )
        )

    expected_index_off = vertex_data_off + total_verts * 6
    if index_data_off != expected_index_off:
        issues.append(
            ValidationIssue(
                "error",
                f"index_data_offset {index_data_off} != expected {expected_index_off}",
            )
        )

    sum_verts = 0
    sum_indices = 0
    off = hull_table_off
    for i in range(hull_count):
        if off + desc_size > len(data):
            issues.append(ValidationIssue("error", f"hull {i}: descriptor truncated"))
            break

        v_off, v_count, i_off, i_count = struct.unpack_from("<IIII", data, off)
        aabb_min = np.array(struct.unpack_from("<3f", data, off + 16), dtype=np.float32)
        aabb_max = np.array(struct.unpack_from("<3f", data, off + 28), dtype=np.float32)

        if np.any(aabb_min > aabb_max):
            issues.append(ValidationIssue("error", f"hull {i}: aabb_min > aabb_max"))

        if v_count < 4:
            issues.append(
                ValidationIssue("warning", f"hull {i}: only {v_count} vertices")
            )

        if i_count % 3 != 0:
            issues.append(
                ValidationIssue(
                    "error", f"hull {i}: index_count {i_count} not divisible by 3"
                )
            )

        if has_bones:
            bone_idx = struct.unpack_from("<i", data, off + 40)[0]
            if bone_idx < -1:
                issues.append(
                    ValidationIssue("error", f"hull {i}: invalid bone_index {bone_idx}")
                )

        i_byte_off = index_data_off + i_off * 2
        if i_byte_off + i_count * 2 <= len(data) and v_count > 0 and i_count > 0:
            indices = np.frombuffer(
                data, dtype=np.uint16, count=i_count, offset=i_byte_off
            )
            if indices.max() >= v_count:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"hull {i}: index {indices.max()} >= vertex_count {v_count}",
                    )
                )

        sum_verts += v_count
        sum_indices += i_count
        off += desc_size

    if sum_verts != total_verts:
        issues.append(
            ValidationIssue(
                "error",
                f"total_vertices {total_verts} != sum {sum_verts}",
            )
        )

    if sum_indices != total_idx:
        issues.append(
            ValidationIssue(
                "error",
                f"total_indices {total_idx} != sum {sum_indices}",
            )
        )

    expected_min = index_data_off + total_idx * 2
    if len(data) < expected_min:
        issues.append(
            ValidationIssue(
                "error",
                f"file truncated: {len(data)} bytes < minimum {expected_min}",
            )
        )

    has_bind_poses = bool(flags & FLAG_HAS_BIND_POSES)
    if has_bind_poses:
        bone_off = expected_min
        if bone_off + 4 > len(data):
            issues.append(
                ValidationIssue("error", "HAS_BIND_POSES set but no bone block")
            )
        else:
            bone_count = struct.unpack_from("<I", data, bone_off)[0]
            bone_off += 4
            for b in range(bone_count):
                if bone_off + 64 > len(data):
                    issues.append(
                        ValidationIssue("error", f"bone {b}: bind_transform truncated")
                    )
                    break
                mat = np.frombuffer(
                    data, dtype=np.float32, count=16, offset=bone_off
                ).reshape(4, 4)
                det = np.linalg.det(mat[:3, :3])
                if abs(det) < 1e-7:
                    issues.append(
                        ValidationIssue(
                            "warning",
                            f"bone {b}: near-singular bind_transform (det={det:.2e})",
                        )
                    )
                bone_off += 64
                if bone_off + 2 > len(data):
                    issues.append(
                        ValidationIssue("error", f"bone {b}: name_length truncated")
                    )
                    break
                name_len = struct.unpack_from("<H", data, bone_off)[0]
                bone_off += 2
                if bone_off + name_len > len(data):
                    issues.append(ValidationIssue("error", f"bone {b}: name truncated"))
                    break
                bone_off += name_len

    return issues
