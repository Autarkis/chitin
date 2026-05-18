# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    concavity: float = 0.05
    opacity_threshold: float = 0.1
    poisson_depth: int = 8
    min_hull_vertices: int = 4
    max_hulls: int = 2048
    opacity_is_logit: bool = False
    coacd_preprocess: bool = True
    coacd_preprocess_resolution: int = 30
