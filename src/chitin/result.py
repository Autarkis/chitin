# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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


@dataclass
class ExtractionResult:
    hulls: list[Hull]
    source_vertex_count: int
    mesh_vertex_count: int

    def to_json(self, path: str | Path) -> None:
        data = {
            "hulls": [
                {
                    "vertices": h.vertices.tolist(),
                    "indices": h.indices.tolist(),
                }
                for h in self.hulls
            ],
            "meta": {
                "hull_count": len(self.hulls),
                "source_vertex_count": self.source_vertex_count,
                "mesh_vertex_count": self.mesh_vertex_count,
            },
        }
        Path(path).write_text(json.dumps(data))

    def to_phys(self, path: str | Path) -> None:
        MAGIC = b"PHYS"
        VERSION = 2
        HEADER_SIZE = 32
        DESCRIPTOR_SIZE = 40

        hull_count = len(self.hulls)
        total_vertices = sum(len(h.vertices) for h in self.hulls)
        total_indices = sum(len(h.indices) for h in self.hulls)

        hull_table_offset = HEADER_SIZE
        vertex_data_offset = hull_table_offset + hull_count * DESCRIPTOR_SIZE
        index_data_offset = vertex_data_offset + total_vertices * 3 * 2  # int16[3]

        with open(path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<H", VERSION))
            f.write(struct.pack("<H", 0))  # flags
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

                vertex_offset += nv
                index_offset += ni

            for hull, (aabb_min, aabb_max) in zip(self.hulls, aabbs):
                quantized = _quantize_vertices(hull.vertices, aabb_min, aabb_max)
                f.write(quantized.tobytes())

            for hull in self.hulls:
                f.write(hull.indices.astype(np.uint16).tobytes())

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
        UsdGeom.Scope.Define(stage, f"/{scene_name}/Colliders")

        for i, hull in enumerate(self.hulls):
            mesh_path = f"/{scene_name}/Colliders/hull_{i:04d}"
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
