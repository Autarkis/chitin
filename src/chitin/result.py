# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
        VERSION = 1
        buf = bytearray()
        buf.extend(MAGIC)
        buf.extend(struct.pack("<I", VERSION))
        buf.extend(struct.pack("<I", len(self.hulls)))
        buf.extend(b"\x00" * 20)  # reserved to 32-byte header

        descriptors = bytearray()
        vertex_buf = bytearray()
        index_buf = bytearray()
        vertex_offset = 0
        index_offset = 0

        for hull in self.hulls:
            nv = len(hull.vertices)
            ni = len(hull.indices)
            descriptors.extend(struct.pack("<I", vertex_offset))
            descriptors.extend(struct.pack("<I", nv))
            descriptors.extend(struct.pack("<I", index_offset))
            descriptors.extend(struct.pack("<I", ni))
            descriptors.extend(b"\x00" * 24)  # pad to 40 bytes per descriptor

            vertex_buf.extend(hull.vertices.astype(np.float32).tobytes())
            index_buf.extend(hull.indices.astype(np.uint32).tobytes())
            vertex_offset += nv
            index_offset += ni

        buf.extend(descriptors)
        buf.extend(vertex_buf)
        buf.extend(index_buf)
        Path(path).write_bytes(bytes(buf))

    def to_usd(self, path: str | Path, scene_name: str = "scene") -> None:
        try:
            from pxr import Sdf, Usd, UsdGeom, UsdPhysics
        except ImportError:
            raise ImportError(
                "USD output requires usd-core. Install with: pip install chitin[usd]"
            )

        stage = Usd.Stage.CreateNew(str(path))
        stage.SetMetadata("upAxis", "Y")
        stage.SetMetadata("metersPerUnit", 1.0)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        root = UsdGeom.Xform.Define(stage, f"/{scene_name}")
        stage.SetDefaultPrim(root.GetPrim())
        colliders_scope = UsdGeom.Scope.Define(stage, f"/{scene_name}/Colliders")

        for i, hull in enumerate(self.hulls):
            mesh_path = f"/{scene_name}/Colliders/hull_{i}"
            mesh = UsdGeom.Mesh.Define(stage, mesh_path)
            mesh.GetPointsAttr().Set([tuple(v) for v in hull.vertices.tolist()])

            face_count = len(hull.indices) // 3
            mesh.GetFaceVertexCountsAttr().Set([3] * face_count)
            mesh.GetFaceVertexIndicesAttr().Set(hull.indices.tolist())

            prim = mesh.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim)
            prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set(
                "convexHull"
            )

        stage.GetRootLayer().Save()
