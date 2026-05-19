# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    concavity: float = 0.05
    opacity_threshold: float = 0.5
    poisson_depth: int = 8
    min_hull_vertices: int = 4
    max_hulls: int = 2048
    opacity_is_logit: bool = False
    coacd_preprocess_mode: str = "auto"
    coacd_preprocess_resolution: int = 50
    max_decompose_vertices: int = 200_000
    lod_concavities: list[float] | None = None
    splat_scale_is_log: bool = True
    splat_surface_ratio: float = 0.2
    spatial_split_threshold: int = 500_000
