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

    if suffix in (".glb", ".gltf", ".fbx"):
        from chitin.adapters.gltf import load_gltf

        return load_gltf(path)

    if suffix in (".usd", ".usda", ".usdc"):
        from chitin.adapters.usd import load_usd

        return load_usd(path)

    raise ValueError(f"Unsupported input format: {suffix}")
