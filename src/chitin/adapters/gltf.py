# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from chitin.adapters import AdapterResult, SkinData
from chitin.gltf_skin import parse_skin


def load_gltf(path: Path) -> AdapterResult:
    fmt = path.suffix.lower().lstrip(".")
    skin_raw = parse_skin(path)
    loaded = trimesh.load(str(path))

    has_skin_weights = (
        skin_raw is not None
        and skin_raw.joint_indices is not None
        and skin_raw.joint_weights is not None
    )

    if has_skin_weights and isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry()
        if isinstance(mesh, trimesh.Trimesh):
            mesh.visual = trimesh.visual.ColorVisuals()
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            skin = SkinData(
                joint_indices=np.asarray(skin_raw.joint_indices, dtype=np.int32),
                joint_weights=np.asarray(skin_raw.joint_weights, dtype=np.float64),
                bone_names=skin_raw.joint_names,
                inverse_bind_matrices=skin_raw.inverse_bind_matrices,
            )
            return AdapterResult(
                positions=verts,
                faces=faces,
                format=fmt,
                skin=skin,
                detected={
                    "is_skinned": True,
                    "bone_count": len(skin_raw.joint_names),
                },
            )

    detected = {"is_skinned": False}

    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry()
        if isinstance(mesh, trimesh.Trimesh):
            mesh.visual = trimesh.visual.ColorVisuals()
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            del mesh
            return AdapterResult(
                positions=verts, faces=faces, format=fmt, detected=detected
            )
        return AdapterResult(
            positions=np.empty((0, 3), dtype=np.float32),
            format=fmt,
            detected=detected,
        )

    loaded.visual = trimesh.visual.ColorVisuals()
    verts = np.asarray(loaded.vertices, dtype=np.float32)
    faces = np.asarray(loaded.faces, dtype=np.int32)
    del loaded
    return AdapterResult(positions=verts, faces=faces, format=fmt, detected=detected)
