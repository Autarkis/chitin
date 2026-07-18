from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"PHYS"
HEADER_SIZE = 32
FLAG_HAS_BONES = 0x01
FLAG_HAS_BIND_POSES = 0x02
FLAG_HAS_LOD = 0x04
KNOWN_FLAGS = FLAG_HAS_BONES | FLAG_HAS_BIND_POSES | FLAG_HAS_LOD
SUPPORTED_VERSIONS = {2, 3}

LOD_TIER_HEADER_SIZE = 24


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
class LodTier:
    concavity: float
    hulls: list[PhysHull]

    @property
    def hull_count(self) -> int:
        return len(self.hulls)

    @property
    def total_vertices(self) -> int:
        return sum(len(h.vertices) for h in self.hulls)

    @property
    def total_indices(self) -> int:
        return sum(len(h.indices) for h in self.hulls)


@dataclass
class PhysFile:
    version: int
    flags: int
    hulls: list[PhysHull]
    bones: list[PhysBone] = field(default_factory=list)
    lod_tiers: list[LodTier] = field(default_factory=list)

    @property
    def has_bones(self) -> bool:
        return bool(self.flags & FLAG_HAS_BONES)

    @property
    def has_bind_poses(self) -> bool:
        return bool(self.flags & FLAG_HAS_BIND_POSES)

    @property
    def has_lod(self) -> bool:
        return bool(self.flags & FLAG_HAS_LOD)

    @property
    def total_vertices(self) -> int:
        return sum(len(h.vertices) for h in self.hulls)

    @property
    def total_indices(self) -> int:
        return sum(len(h.indices) for h in self.hulls)

    @property
    def total_triangles(self) -> int:
        return self.total_indices // 3

    def lod_tier(self, concavity: float) -> LodTier | None:
        if not self.lod_tiers:
            return None
        return min(self.lod_tiers, key=lambda t: abs(t.concavity - concavity))


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

    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"unsupported .phys version {version} "
            f"(supported: {sorted(SUPPORTED_VERSIONS)})"
        )

    unknown_flags = flags & ~KNOWN_FLAGS
    if unknown_flags:
        raise ValueError(f"unknown flags 0x{unknown_flags:04x} in .phys header")

    has_bones = bool(flags & FLAG_HAS_BONES)
    desc_size = _descriptor_size(flags)

    if hull_table_off != HEADER_SIZE:
        raise ValueError(
            f"hull_table_offset {hull_table_off} != expected {HEADER_SIZE}"
        )

    expected_vertex_off = hull_table_off + hull_count * desc_size
    if vertex_data_off != expected_vertex_off:
        raise ValueError(
            f"vertex_data_offset {vertex_data_off} != expected {expected_vertex_off}"
        )

    expected_index_off = vertex_data_off + total_verts * 6
    if index_data_off != expected_index_off:
        raise ValueError(
            f"index_data_offset {index_data_off} != expected {expected_index_off}"
        )

    expected_min = index_data_off + total_idx * 2
    if len(data) < expected_min:
        raise ValueError(f"file truncated: {len(data)} bytes < minimum {expected_min}")

    # Peek the bone count (bind-pose block trails index data) to bounds-check
    # per-hull bone indices while reading.
    bone_table_count = None
    if flags & FLAG_HAS_BIND_POSES and expected_min + 4 <= len(data):
        bone_table_count = struct.unpack_from("<I", data, expected_min)[0]

    hulls: list[PhysHull] = []
    off = hull_table_off
    expected_v_off = 0
    expected_i_off = 0
    for i in range(hull_count):
        v_off, v_count, i_off, i_count = struct.unpack_from("<IIII", data, off)
        if v_off + v_count > total_verts:
            raise ValueError(
                f"hull {i}: vertex range [{v_off}, {v_off + v_count}) "
                f"exceeds total_vertices {total_verts}"
            )
        if i_off + i_count > total_idx:
            raise ValueError(
                f"hull {i}: index range [{i_off}, {i_off + i_count}) "
                f"exceeds total_indices {total_idx}"
            )
        if v_off != expected_v_off:
            raise ValueError(
                f"hull {i}: vertex_offset {v_off} != expected {expected_v_off} "
                "(non-contiguous or overlapping range)"
            )
        if i_off != expected_i_off:
            raise ValueError(
                f"hull {i}: index_offset {i_off} != expected {expected_i_off} "
                "(non-contiguous or overlapping range)"
            )
        aabb_min = np.array(struct.unpack_from("<3f", data, off + 16), dtype=np.float32)
        aabb_max = np.array(struct.unpack_from("<3f", data, off + 28), dtype=np.float32)
        if not np.all(np.isfinite(aabb_min)) or not np.all(np.isfinite(aabb_max)):
            raise ValueError(f"hull {i}: non-finite aabb")
        bone_idx = None
        if has_bones:
            raw_bone = struct.unpack_from("<i", data, off + 40)[0]
            if raw_bone < -1:
                raise ValueError(f"hull {i}: invalid bone_index {raw_bone}")
            if bone_table_count is not None and raw_bone >= bone_table_count:
                raise ValueError(
                    f"hull {i}: bone_index {raw_bone} >= bone_count {bone_table_count}"
                )
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
        expected_v_off += v_count
        expected_i_off += i_count

    bones: list[PhysBone] = []
    next_block_off = index_data_off + total_idx * 2
    if flags & FLAG_HAS_BIND_POSES:
        bone_off = next_block_off
        bone_count = struct.unpack_from("<I", data, bone_off)[0]
        bone_off += 4
        for _ in range(bone_count):
            mat = (
                np.frombuffer(data, dtype=np.float32, count=16, offset=bone_off)
                .reshape(4, 4)
                .copy()
            )
            if not np.all(np.isfinite(mat)):
                raise ValueError("non-finite bind_transform")
            bone_off += 64
            name_len = struct.unpack_from("<H", data, bone_off)[0]
            bone_off += 2
            name = data[bone_off : bone_off + name_len].decode("utf-8")
            bone_off += name_len
            bones.append(PhysBone(name=name, bind_transform=mat))
        next_block_off = bone_off

    lod_tiers: list[LodTier] = []
    if flags & FLAG_HAS_LOD:
        lod_off = next_block_off

        tier_count = struct.unpack_from("<I", data, lod_off)[0]
        lod_off += 4

        for _ in range(tier_count):
            t_concavity = struct.unpack_from("<f", data, lod_off)[0]
            t_hull_count = struct.unpack_from("<I", data, lod_off + 4)[0]
            t_total_verts = struct.unpack_from("<I", data, lod_off + 8)[0]
            t_total_idx = struct.unpack_from("<I", data, lod_off + 12)[0]
            t_data_size = struct.unpack_from("<I", data, lod_off + 16)[0]
            lod_off += LOD_TIER_HEADER_SIZE

            t_hull_table_off = lod_off
            t_vertex_data_off = t_hull_table_off + t_hull_count * desc_size
            t_index_data_off = t_vertex_data_off + t_total_verts * 6

            tier_hulls: list[PhysHull] = []
            off = t_hull_table_off
            t_expected_v_off = 0
            t_expected_i_off = 0
            for _ in range(t_hull_count):
                v_off, v_count, i_off, i_count = struct.unpack_from("<IIII", data, off)
                if v_off + v_count > t_total_verts:
                    raise ValueError("lod tier hull: vertex range exceeds tier total")
                if i_off + i_count > t_total_idx:
                    raise ValueError("lod tier hull: index range exceeds tier total")
                if v_off != t_expected_v_off or i_off != t_expected_i_off:
                    raise ValueError(
                        "lod tier hull: non-contiguous or overlapping range"
                    )
                aabb_min = np.array(
                    struct.unpack_from("<3f", data, off + 16), dtype=np.float32
                )
                aabb_max = np.array(
                    struct.unpack_from("<3f", data, off + 28), dtype=np.float32
                )
                if not np.all(np.isfinite(aabb_min)) or not np.all(
                    np.isfinite(aabb_max)
                ):
                    raise ValueError("lod tier hull: non-finite aabb")
                bone_idx = None
                if has_bones:
                    raw_bone = struct.unpack_from("<i", data, off + 40)[0]
                    if raw_bone < -1:
                        raise ValueError(
                            f"lod tier hull: invalid bone_index {raw_bone}"
                        )
                    if bone_table_count is not None and raw_bone >= bone_table_count:
                        raise ValueError(
                            f"lod tier hull: bone_index {raw_bone} >= "
                            f"bone_count {bone_table_count}"
                        )
                    bone_idx = None if raw_bone == -1 else raw_bone
                off += desc_size

                v_byte_off = t_vertex_data_off + v_off * 6
                raw = np.frombuffer(
                    data, dtype=np.int16, count=v_count * 3, offset=v_byte_off
                )
                quantized = raw.reshape(-1, 3).astype(np.float32)
                extent = aabb_max - aabb_min
                extent = np.where(extent == 0, 1.0, extent)
                vertices = (quantized + 32768) / 65535 * extent + aabb_min

                i_byte_off = t_index_data_off + i_off * 2
                indices = np.frombuffer(
                    data, dtype=np.uint16, count=i_count, offset=i_byte_off
                ).copy()

                tier_hulls.append(
                    PhysHull(
                        vertices=vertices,
                        indices=indices,
                        aabb_min=aabb_min,
                        aabb_max=aabb_max,
                        bone_index=bone_idx,
                    )
                )
                t_expected_v_off += v_count
                t_expected_i_off += i_count

            lod_tiers.append(LodTier(concavity=t_concavity, hulls=tier_hulls))
            lod_off += t_data_size
        next_block_off = lod_off

    if len(data) > next_block_off:
        raise ValueError(
            f"{len(data) - next_block_off} trailing bytes after end of data"
        )

    return PhysFile(
        version=version, flags=flags, hulls=hulls, bones=bones, lod_tiers=lod_tiers
    )


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

    if version not in SUPPORTED_VERSIONS:
        issues.append(
            ValidationIssue(
                "error",
                f"unsupported version {version} "
                f"(supported: {sorted(SUPPORTED_VERSIONS)})",
            )
        )
        return issues

    unknown_flags = flags & ~KNOWN_FLAGS
    if unknown_flags:
        issues.append(ValidationIssue("error", f"unknown flags 0x{unknown_flags:04x}"))

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

    # Peek the bone count (the bind-pose block trails the index data) so each
    # hull's bone_index can be range-checked during the loop below.
    bone_table_count: int | None = None
    if flags & FLAG_HAS_BIND_POSES:
        bone_block_off = index_data_off + total_idx * 2
        if bone_block_off + 4 <= len(data):
            bone_table_count = struct.unpack_from("<I", data, bone_block_off)[0]

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

        if not np.all(np.isfinite(aabb_min)) or not np.all(np.isfinite(aabb_max)):
            issues.append(ValidationIssue("error", f"hull {i}: non-finite aabb"))
        elif np.any(aabb_min > aabb_max):
            issues.append(ValidationIssue("error", f"hull {i}: aabb_min > aabb_max"))

        # Hull ranges are contiguous and non-overlapping: each offset must equal
        # the running total of preceding counts (the writer packs sequentially).
        if v_off != sum_verts:
            issues.append(
                ValidationIssue(
                    "error",
                    f"hull {i}: vertex_offset {v_off} != expected {sum_verts} "
                    "(non-contiguous or overlapping range)",
                )
            )
        if i_off != sum_indices:
            issues.append(
                ValidationIssue(
                    "error",
                    f"hull {i}: index_offset {i_off} != expected {sum_indices} "
                    "(non-contiguous or overlapping range)",
                )
            )

        # Each hull's declared vertex/index range must stay within the arrays.
        if v_off + v_count > total_verts:
            issues.append(
                ValidationIssue(
                    "error",
                    f"hull {i}: vertex range [{v_off}, {v_off + v_count}) "
                    f"exceeds total_vertices {total_verts}",
                )
            )
        if i_off + i_count > total_idx:
            issues.append(
                ValidationIssue(
                    "error",
                    f"hull {i}: index range [{i_off}, {i_off + i_count}) "
                    f"exceeds total_indices {total_idx}",
                )
            )

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
            elif bone_table_count is not None and bone_idx >= bone_table_count:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"hull {i}: bone_index {bone_idx} >= "
                        f"bone_count {bone_table_count}",
                    )
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

    next_block_off = expected_min
    has_bind_poses = bool(flags & FLAG_HAS_BIND_POSES)
    if has_bind_poses:
        bone_off = next_block_off
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
                if not np.all(np.isfinite(mat)):
                    issues.append(
                        ValidationIssue("error", f"bone {b}: non-finite bind_transform")
                    )
                else:
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
            else:
                next_block_off = bone_off

    has_lod = bool(flags & FLAG_HAS_LOD)
    if has_lod:
        lod_off = next_block_off
        if lod_off + 4 > len(data):
            issues.append(ValidationIssue("error", "HAS_LOD set but no LOD block"))
            return issues

        tier_count = struct.unpack_from("<I", data, lod_off)[0]
        lod_off += 4

        prev_concavity = 0.0
        for t in range(tier_count):
            if lod_off + LOD_TIER_HEADER_SIZE > len(data):
                issues.append(
                    ValidationIssue("error", f"LOD tier {t}: header truncated")
                )
                break

            t_concavity = struct.unpack_from("<f", data, lod_off)[0]
            t_hull_count = struct.unpack_from("<I", data, lod_off + 4)[0]
            t_total_verts = struct.unpack_from("<I", data, lod_off + 8)[0]
            t_total_idx = struct.unpack_from("<I", data, lod_off + 12)[0]
            t_data_size = struct.unpack_from("<I", data, lod_off + 16)[0]
            lod_off += LOD_TIER_HEADER_SIZE

            if t_concavity <= 0:
                issues.append(
                    ValidationIssue(
                        "warning",
                        f"LOD tier {t}: non-positive concavity {t_concavity}",
                    )
                )
            if t_concavity < prev_concavity:
                issues.append(
                    ValidationIssue(
                        "warning",
                        f"LOD tier {t}: concavity {t_concavity} < previous {prev_concavity} (not ascending)",
                    )
                )
            prev_concavity = t_concavity

            expected_data_size = (
                t_hull_count * desc_size + t_total_verts * 6 + t_total_idx * 2
            )
            if t_data_size != expected_data_size:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"LOD tier {t}: data_size {t_data_size} != expected {expected_data_size}",
                    )
                )

            if lod_off + t_data_size > len(data):
                issues.append(ValidationIssue("error", f"LOD tier {t}: data truncated"))
                break

            t_hull_table_off = lod_off
            t_vertex_data_off = t_hull_table_off + t_hull_count * desc_size
            t_index_data_off = t_vertex_data_off + t_total_verts * 6

            t_sum_verts = 0
            t_sum_idx = 0
            off = t_hull_table_off
            for h in range(t_hull_count):
                if off + desc_size > len(data):
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"LOD tier {t} hull {h}: descriptor truncated",
                        )
                    )
                    break

                v_off, v_count, i_off, i_count = struct.unpack_from("<IIII", data, off)
                t_aabb_min = np.array(
                    struct.unpack_from("<3f", data, off + 16), dtype=np.float32
                )
                t_aabb_max = np.array(
                    struct.unpack_from("<3f", data, off + 28), dtype=np.float32
                )

                if not np.all(np.isfinite(t_aabb_min)) or not np.all(
                    np.isfinite(t_aabb_max)
                ):
                    issues.append(
                        ValidationIssue(
                            "error", f"LOD tier {t} hull {h}: non-finite aabb"
                        )
                    )
                elif np.any(t_aabb_min > t_aabb_max):
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"LOD tier {t} hull {h}: aabb_min > aabb_max",
                        )
                    )

                if v_off + v_count > t_total_verts:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"LOD tier {t} hull {h}: vertex range exceeds tier "
                            f"total_vertices {t_total_verts}",
                        )
                    )
                if i_off + i_count > t_total_idx:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"LOD tier {t} hull {h}: index range exceeds tier "
                            f"total_indices {t_total_idx}",
                        )
                    )
                if v_off != t_sum_verts:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"LOD tier {t} hull {h}: vertex_offset {v_off} != "
                            f"expected {t_sum_verts} (non-contiguous or overlapping)",
                        )
                    )
                if i_off != t_sum_idx:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"LOD tier {t} hull {h}: index_offset {i_off} != "
                            f"expected {t_sum_idx} (non-contiguous or overlapping)",
                        )
                    )

                if v_count < 4:
                    issues.append(
                        ValidationIssue(
                            "warning",
                            f"LOD tier {t} hull {h}: only {v_count} vertices",
                        )
                    )

                if i_count % 3 != 0:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"LOD tier {t} hull {h}: index_count {i_count} not divisible by 3",
                        )
                    )

                if has_bones:
                    bone_idx = struct.unpack_from("<i", data, off + 40)[0]
                    if bone_idx < -1:
                        issues.append(
                            ValidationIssue(
                                "error",
                                f"LOD tier {t} hull {h}: invalid bone_index {bone_idx}",
                            )
                        )
                    elif bone_table_count is not None and bone_idx >= bone_table_count:
                        issues.append(
                            ValidationIssue(
                                "error",
                                f"LOD tier {t} hull {h}: bone_index {bone_idx} >= "
                                f"bone_count {bone_table_count}",
                            )
                        )

                i_byte_off = t_index_data_off + i_off * 2
                if (
                    i_byte_off + i_count * 2 <= len(data)
                    and v_count > 0
                    and i_count > 0
                ):
                    indices = np.frombuffer(
                        data, dtype=np.uint16, count=i_count, offset=i_byte_off
                    )
                    if indices.max() >= v_count:
                        issues.append(
                            ValidationIssue(
                                "error",
                                f"LOD tier {t} hull {h}: index {indices.max()} >= vertex_count {v_count}",
                            )
                        )

                t_sum_verts += v_count
                t_sum_idx += i_count
                off += desc_size

            if t_sum_verts != t_total_verts:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"LOD tier {t}: total_vertices {t_total_verts} != sum {t_sum_verts}",
                    )
                )
            if t_sum_idx != t_total_idx:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"LOD tier {t}: total_indices {t_total_idx} != sum {t_sum_idx}",
                    )
                )

            lod_off += t_data_size

        next_block_off = lod_off

    if len(data) > next_block_off:
        trailing = len(data) - next_block_off
        issues.append(
            ValidationIssue(
                "error",
                f"{trailing} trailing bytes after end of data",
            )
        )

    return issues
