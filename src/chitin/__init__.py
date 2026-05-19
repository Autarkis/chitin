# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
"""Convex collision geometry from point clouds, meshes, and gaussian splats."""

from chitin.config import Config
from chitin.phys import LodTier, PhysBone, PhysFile, PhysHull, read_phys, validate_phys
from chitin.plan import BuildPlan
from chitin.result import BoneInfo, ExtractionResult


def __getattr__(name):
    if name in (
        "extract",
        "extract_from_arrays",
        "extract_from_mesh",
        "extract_from_rigged_mesh",
    ):
        from chitin import core

        return getattr(core, name)
    raise AttributeError(f"module 'chitin' has no attribute {name!r}")


__all__ = [
    "BoneInfo",
    "BuildPlan",
    "Config",
    "ExtractionResult",
    "LodTier",
    "PhysBone",
    "PhysFile",
    "PhysHull",
    "extract",
    "extract_from_arrays",
    "extract_from_mesh",
    "extract_from_rigged_mesh",
    "read_phys",
    "validate_phys",
]
