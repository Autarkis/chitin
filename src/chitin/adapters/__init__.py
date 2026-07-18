from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SkinData:
    joint_indices: np.ndarray
    joint_weights: np.ndarray
    bone_names: list[str] | None = None
    inverse_bind_matrices: dict[int, np.ndarray] | None = None


@dataclass
class AdapterResult:
    positions: np.ndarray
    format: str
    faces: np.ndarray | None = None
    normals: np.ndarray | None = None
    scales: np.ndarray | None = None
    rots: np.ndarray | None = None
    opacity: np.ndarray | None = None
    skin: SkinData | None = None
    detected: dict[str, object] = field(default_factory=dict)


def load(path: Path) -> AdapterResult:
    suffix = path.suffix.lower()

    if suffix == ".ply":
        from chitin.adapters.ply import load_ply

        return load_ply(path)

    if suffix in (".obj", ".stl", ".off"):
        from chitin.adapters.mesh import load_mesh

        return load_mesh(path)

    if suffix in (".glb", ".gltf"):
        from chitin.adapters.gltf import load_gltf

        return load_gltf(path)

    if suffix == ".fbx":
        # trimesh has no FBX loader, so transparently convert to GLB via Blender
        # (into a temp dir) and load that. Blender must be on PATH.
        import tempfile

        from chitin.adapters.gltf import load_gltf
        from chitin.convert import convert_fbx_to_glb

        with tempfile.TemporaryDirectory() as td:
            glb_path = Path(td) / (path.stem + ".glb")
            convert_fbx_to_glb(path, glb_path)
            result = load_gltf(glb_path)
        result.detected["converted_from_fbx"] = True
        return result

    if suffix in (".usd", ".usda", ".usdc"):
        from chitin.adapters.usd import load_usd

        return load_usd(path)

    raise ValueError(f"Unsupported input format: {suffix}")
