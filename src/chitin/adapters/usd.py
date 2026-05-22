from __future__ import annotations

from pathlib import Path

import numpy as np

from chitin.adapters import AdapterResult


def load_usd(path: Path) -> AdapterResult:
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        raise ImportError(
            "USD input requires usd-core. Install with: pip install chitin[usd]"
        )

    fmt = path.suffix.lower().lstrip(".")
    stage = Usd.Stage.Open(str(path))
    time_code = Usd.TimeCode.Default()
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    mesh_count = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        raw_points = mesh.GetPointsAttr().Get(time_code)
        if raw_points is None or len(raw_points) == 0:
            continue

        mesh_count += 1
        points = np.array(raw_points, dtype=np.float64)

        xformable = UsdGeom.Xformable(prim)
        world_xform = xformable.ComputeLocalToWorldTransform(time_code)
        mat = np.array(world_xform, dtype=np.float64)
        if not np.allclose(mat, np.eye(4)):
            ones = np.ones((len(points), 1), dtype=np.float64)
            homogeneous = np.hstack([points, ones])
            points = (homogeneous @ mat)[:, :3]

        face_counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        face_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

        tris = []
        idx = 0
        for count in face_counts:
            if count == 3:
                tris.append(face_indices[idx : idx + 3] + vertex_offset)
            elif count > 3:
                for j in range(1, count - 1):
                    tris.append(
                        np.array(
                            [
                                face_indices[idx] + vertex_offset,
                                face_indices[idx + j] + vertex_offset,
                                face_indices[idx + j + 1] + vertex_offset,
                            ],
                            dtype=np.int32,
                        )
                    )
            idx += count

        all_vertices.append(points)
        if tris:
            all_faces.append(np.array(tris, dtype=np.int32))
        vertex_offset += len(points)

    detected = {"mesh_prim_count": mesh_count}

    if not all_vertices or not all_faces:
        return AdapterResult(
            positions=np.empty((0, 3), dtype=np.float64),
            format=fmt,
            detected=detected,
        )

    vertices = np.concatenate(all_vertices)
    faces = np.concatenate(all_faces)
    return AdapterResult(positions=vertices, faces=faces, format=fmt, detected=detected)
