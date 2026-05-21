# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chitin.plan import BuildPlan


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
class LodHulls:
    concavity: float
    hulls: list[Hull]


@dataclass
class ExtractionResult:
    hulls: list[Hull]
    source_vertex_count: int
    mesh_vertex_count: int
    bones: list[BoneInfo] | None = None
    build_plan: BuildPlan | None = None
    lod_tiers: list[LodHulls] | None = None

    def to_json(self, path: str | Path) -> None:
        from chitin.exporters.json import export_json

        export_json(self, path)

    def to_phys(self, path: str | Path) -> None:
        from chitin.exporters.phys import export_phys

        export_phys(self, path)

    def to_usd(self, path: str | Path, scene_name: str = "scene") -> None:
        from chitin.exporters.usd import export_usd

        export_usd(self, path, scene_name=scene_name)
