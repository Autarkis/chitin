# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from chitin.adapters import AdapterResult


def load_mesh(path: Path) -> AdapterResult:
    fmt = path.suffix.lower().lstrip(".")
    mesh = trimesh.load(str(path), force="mesh")
    mesh.visual = trimesh.visual.ColorVisuals()
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    del mesh
    return AdapterResult(positions=verts, faces=faces, format=fmt)
