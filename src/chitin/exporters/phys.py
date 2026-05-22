from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from chitin.result import ExtractionResult


def _quantize_vertices(
    vertices: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray
) -> np.ndarray:
    extent = aabb_max - aabb_min
    extent = np.where(extent == 0, 1.0, extent)
    normalized = (vertices - aabb_min) / extent
    quantized = np.clip(normalized * 65535 - 32768, -32768, 32767)
    return quantized.astype(np.int16)


def export_phys(result: ExtractionResult, path: str | Path) -> None:
    MAGIC = b"PHYS"
    HEADER_SIZE = 32
    FLAG_HAS_BONES = 0x01
    FLAG_HAS_BIND_POSES = 0x02
    FLAG_HAS_LOD = 0x04

    has_bones = any(h.bone_index is not None for h in result.hulls)
    has_bind_poses = result.bones is not None and len(result.bones) > 0
    has_lod = result.lod_tiers is not None and len(result.lod_tiers) > 0
    flags = 0
    if has_bones:
        flags |= FLAG_HAS_BONES
    if has_bind_poses:
        flags |= FLAG_HAS_BIND_POSES
    if has_lod:
        flags |= FLAG_HAS_LOD
    version = 3
    descriptor_size = 44 if has_bones else 40

    all_hull_lists = [(result.hulls, "LOD0")]
    if has_lod:
        for t, tier in enumerate(result.lod_tiers):
            all_hull_lists.append((tier.hulls, f"LOD tier {t}"))
    for hull_list, label in all_hull_lists:
        for i, h in enumerate(hull_list):
            if len(h.vertices) > 65535:
                raise ValueError(
                    f"{label} hull {i}: {len(h.vertices)} vertices exceeds uint16 limit (65535)"
                )
            if len(h.indices) > 0 and h.indices.max() > 65535:
                raise ValueError(
                    f"{label} hull {i}: index value {h.indices.max()} exceeds uint16 limit"
                )

    hull_count = len(result.hulls)
    total_vertices = sum(len(h.vertices) for h in result.hulls)
    total_indices = sum(len(h.indices) for h in result.hulls)

    hull_table_offset = HEADER_SIZE
    vertex_data_offset = hull_table_offset + hull_count * descriptor_size
    index_data_offset = vertex_data_offset + total_vertices * 3 * 2

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<H", version))
        f.write(struct.pack("<H", flags))
        f.write(struct.pack("<I", hull_count))
        f.write(struct.pack("<I", total_vertices))
        f.write(struct.pack("<I", total_indices))
        f.write(struct.pack("<I", hull_table_offset))
        f.write(struct.pack("<I", vertex_data_offset))
        f.write(struct.pack("<I", index_data_offset))

        vertex_offset = 0
        index_offset = 0
        aabbs = []
        for hull in result.hulls:
            nv = len(hull.vertices)
            ni = len(hull.indices)
            aabb_min = hull.vertices.min(axis=0).astype(np.float32)
            aabb_max = hull.vertices.max(axis=0).astype(np.float32)
            aabbs.append((aabb_min, aabb_max))

            f.write(struct.pack("<I", vertex_offset))
            f.write(struct.pack("<I", nv))
            f.write(struct.pack("<I", index_offset))
            f.write(struct.pack("<I", ni))
            f.write(struct.pack("<3f", *aabb_min))
            f.write(struct.pack("<3f", *aabb_max))
            if has_bones:
                f.write(
                    struct.pack(
                        "<i", hull.bone_index if hull.bone_index is not None else -1
                    )
                )

            vertex_offset += nv
            index_offset += ni

        for hull, (aabb_min, aabb_max) in zip(result.hulls, aabbs):
            quantized = _quantize_vertices(hull.vertices, aabb_min, aabb_max)
            f.write(quantized.tobytes())

        for hull in result.hulls:
            f.write(hull.indices.astype(np.uint16).tobytes())

        if has_bind_poses:
            f.write(struct.pack("<I", len(result.bones)))
            for bone in result.bones:
                mat = bone.bind_transform.astype(np.float32)
                f.write(mat.tobytes())
                name_bytes = bone.name.encode("utf-8")
                f.write(struct.pack("<H", len(name_bytes)))
                f.write(name_bytes)

        if has_lod:
            f.write(struct.pack("<I", len(result.lod_tiers)))
            for tier in result.lod_tiers:
                t_hull_count = len(tier.hulls)
                t_total_verts = sum(len(h.vertices) for h in tier.hulls)
                t_total_idx = sum(len(h.indices) for h in tier.hulls)
                t_data_size = (
                    t_hull_count * descriptor_size + t_total_verts * 6 + t_total_idx * 2
                )
                f.write(struct.pack("<f", tier.concavity))
                f.write(struct.pack("<I", t_hull_count))
                f.write(struct.pack("<I", t_total_verts))
                f.write(struct.pack("<I", t_total_idx))
                f.write(struct.pack("<I", t_data_size))
                f.write(struct.pack("<I", 0))  # reserved

                t_vertex_offset = 0
                t_index_offset = 0
                t_aabbs = []
                for hull in tier.hulls:
                    nv = len(hull.vertices)
                    ni = len(hull.indices)
                    t_aabb_min = hull.vertices.min(axis=0).astype(np.float32)
                    t_aabb_max = hull.vertices.max(axis=0).astype(np.float32)
                    t_aabbs.append((t_aabb_min, t_aabb_max))
                    f.write(struct.pack("<I", t_vertex_offset))
                    f.write(struct.pack("<I", nv))
                    f.write(struct.pack("<I", t_index_offset))
                    f.write(struct.pack("<I", ni))
                    f.write(struct.pack("<3f", *t_aabb_min))
                    f.write(struct.pack("<3f", *t_aabb_max))
                    if has_bones:
                        f.write(
                            struct.pack(
                                "<i",
                                hull.bone_index if hull.bone_index is not None else -1,
                            )
                        )
                    t_vertex_offset += nv
                    t_index_offset += ni

                for hull, (t_aabb_min, t_aabb_max) in zip(tier.hulls, t_aabbs):
                    quantized = _quantize_vertices(
                        hull.vertices, t_aabb_min, t_aabb_max
                    )
                    f.write(quantized.tobytes())

                for hull in tier.hulls:
                    f.write(hull.indices.astype(np.uint16).tobytes())
