"""Convex collision geometry from point clouds, meshes, and gaussian splats."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("chitin")
except PackageNotFoundError:
    __version__ = "0.1.0"

from chitin.analyze import InputAnalysis, analyze_arrays, analyze_input
from chitin.config import Config
from chitin.phys import LodTier, PhysBone, PhysFile, PhysHull, read_phys, validate_phys
from chitin.plan import BuildPlan
from chitin.resolve import ResolvedConfig, resolve_config
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
    "__version__",
    "BoneInfo",
    "BuildPlan",
    "Config",
    "ExtractionResult",
    "InputAnalysis",
    "LodTier",
    "PhysBone",
    "PhysFile",
    "PhysHull",
    "ResolvedConfig",
    "analyze_arrays",
    "analyze_input",
    "extract",
    "extract_from_arrays",
    "extract_from_mesh",
    "extract_from_rigged_mesh",
    "read_phys",
    "resolve_config",
    "validate_phys",
]
