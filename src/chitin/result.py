# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chitin.plan import BuildPlan


def _quantize_vertices(
    vertices: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray
) -> np.ndarray:
    extent = aabb_max - aabb_min
    extent = np.where(extent == 0, 1.0, extent)
    normalized = (vertices - aabb_min) / extent
    quantized = np.clip(normalized * 65535 - 32768, -32768, 32767)
    return quantized.astype(np.int16)


@dataclass
class Hull:
    vertices: np.ndarray  # (N, 3) float32
    indices: np.ndarray  # (M,) uint32, triangle indices
    bone_name: str | None = None
    bone_index: int | None = None


@dataclass
class BoneInfo:
    name: str
    index: int
    bind_transform: np.ndarray  # (4, 4) float64, world-space bind pose


@dataclass
class ExtractionResult:
    hulls: list[Hull]
    source_vertex_count: int
    mesh_vertex_count: int
    bones: list[BoneInfo] | None = None
    build_plan: BuildPlan | None = None

    def to_json(self, path: str | Path) -> None:
        hull_dicts = []
        for h in self.hulls:
            d = {
                "vertices": h.vertices.tolist(),
                "indices": h.indices.tolist(),
            }
            if h.bone_name is not None:
                d["bone_name"] = h.bone_name
            if h.bone_index is not None:
                d["bone_index"] = h.bone_index
            hull_dicts.append(d)

        meta = {
            "hull_count": len(self.hulls),
            "source_vertex_count": self.source_vertex_count,
            "mesh_vertex_count": self.mesh_vertex_count,
        }
        if self.bones:
            meta["rigged"] = True
            meta["bones"] = [
                {
                    "name": b.name,
                    "index": b.index,
                    "bind_transform": b.bind_transform.tolist(),
                }
                for b in self.bones
            ]

        data = {"hulls": hull_dicts, "meta": meta}
        Path(path).write_text(json.dumps(data))

    def to_phys(self, path: str | Path) -> None:
        MAGIC = b"PHYS"
        VERSION = 2
        HEADER_SIZE = 32
        FLAG_HAS_BONES = 0x01
        FLAG_HAS_BIND_POSES = 0x02

        has_bones = any(h.bone_index is not None for h in self.hulls)
        has_bind_poses = self.bones is not None and len(self.bones) > 0
        flags = 0
        if has_bones:
            flags |= FLAG_HAS_BONES
        if has_bind_poses:
            flags |= FLAG_HAS_BIND_POSES
        descriptor_size = 44 if has_bones else 40

        for i, h in enumerate(self.hulls):
            if len(h.vertices) > 65535:
                raise ValueError(
                    f"hull {i}: {len(h.vertices)} vertices exceeds uint16 limit (65535)"
                )
            if len(h.indices) > 0 and h.indices.max() > 65535:
                raise ValueError(
                    f"hull {i}: index value {h.indices.max()} exceeds uint16 limit"
                )

        hull_count = len(self.hulls)
        total_vertices = sum(len(h.vertices) for h in self.hulls)
        total_indices = sum(len(h.indices) for h in self.hulls)

        hull_table_offset = HEADER_SIZE
        vertex_data_offset = hull_table_offset + hull_count * descriptor_size
        index_data_offset = vertex_data_offset + total_vertices * 3 * 2

        with open(path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<H", VERSION))
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
            for hull in self.hulls:
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

            for hull, (aabb_min, aabb_max) in zip(self.hulls, aabbs):
                quantized = _quantize_vertices(hull.vertices, aabb_min, aabb_max)
                f.write(quantized.tobytes())

            for hull in self.hulls:
                f.write(hull.indices.astype(np.uint16).tobytes())

            if has_bind_poses:
                f.write(struct.pack("<I", len(self.bones)))
                for bone in self.bones:
                    mat = bone.bind_transform.astype(np.float32)
                    f.write(mat.tobytes())
                    name_bytes = bone.name.encode("utf-8")
                    f.write(struct.pack("<H", len(name_bytes)))
                    f.write(name_bytes)

    def to_usd(self, path: str | Path, scene_name: str = "scene") -> None:
        try:
            from pxr import Gf, Usd, UsdGeom, UsdPhysics
        except ImportError:
            raise ImportError(
                "USD output requires usd-core. Install with: pip install chitin[usd]"
            )

        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        root = UsdGeom.Xform.Define(stage, f"/{scene_name}")
        stage.SetDefaultPrim(root.GetPrim())
        colliders_path = f"/{scene_name}/Colliders"
        UsdGeom.Scope.Define(stage, colliders_path)

        bone_xforms_by_name: dict[str, np.ndarray] = {}
        if self.bones:
            for b in self.bones:
                safe = b.name.replace("/", "_").replace(" ", "_")
                bone_xforms_by_name[safe] = b.bind_transform

        created_scopes: set[str] = set()
        bone_counters: dict[str, int] = {}

        for i, hull in enumerate(self.hulls):
            if hull.bone_name is not None:
                safe_bone = hull.bone_name.replace("/", "_").replace(" ", "_")
                scope_path = f"{colliders_path}/{safe_bone}"
                if scope_path not in created_scopes:
                    xform = UsdGeom.Xform.Define(stage, scope_path)
                    if safe_bone in bone_xforms_by_name:
                        mat = bone_xforms_by_name[safe_bone]
                        gf_mat = Gf.Matrix4d(*mat.flatten().tolist())
                        xform.AddTransformOp().Set(gf_mat)
                    created_scopes.add(scope_path)
                idx = bone_counters.get(safe_bone, 0)
                bone_counters[safe_bone] = idx + 1
                mesh_path = f"{scope_path}/hull_{idx:04d}"
            else:
                mesh_path = f"{colliders_path}/hull_{i:04d}"

            mesh = UsdGeom.Mesh.Define(stage, mesh_path)

            points = [
                Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in hull.vertices
            ]
            mesh.CreatePointsAttr().Set(points)
            mesh.CreateFaceVertexCountsAttr().Set([3] * (len(hull.indices) // 3))
            mesh.CreateFaceVertexIndicesAttr().Set(hull.indices.tolist())

            prim = mesh.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_col.CreateApproximationAttr().Set("convexHull")

        stage.GetRootLayer().Save()
