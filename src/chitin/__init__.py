# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
"""Convex collision geometry from point clouds, meshes, and gaussian splats."""

from chitin.config import Config
from chitin.core import extract, extract_from_arrays, extract_from_mesh
from chitin.result import ExtractionResult

__all__ = [
    "Config",
    "ExtractionResult",
    "extract",
    "extract_from_arrays",
    "extract_from_mesh",
]
