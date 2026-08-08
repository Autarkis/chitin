"""Convex collision geometry from point clouds, meshes, and gaussian splats."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("chitin")
except PackageNotFoundError:
    __version__ = "0.1.0"

from chitin.acceptance import (
    PROFILES,
    AcceptancePolicy,
    Profile,
    Verdict,
    apply_profile,
    evaluate,
    get_profile,
)
from chitin.analyze import InputAnalysis, analyze_arrays, analyze_input
from chitin.config import Config
from chitin.manifest import build_manifest, verify_bundle, write_manifest
from chitin.phys import LodTier, PhysBone, PhysFile, PhysHull, read_phys, validate_phys
from chitin.plan import BuildPlan
from chitin.report import (
    REPORT_VERSION,
    CompilationReport,
    ReportMetric,
    ReportWarning,
    build_compilation_report,
    validate_compilation_report,
)
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
    "PROFILES",
    "AcceptancePolicy",
    "BoneInfo",
    "BuildPlan",
    "CompilationReport",
    "Config",
    "ExtractionResult",
    "InputAnalysis",
    "LodTier",
    "PhysBone",
    "PhysFile",
    "PhysHull",
    "Profile",
    "REPORT_VERSION",
    "ReportMetric",
    "ReportWarning",
    "ResolvedConfig",
    "Verdict",
    "analyze_arrays",
    "analyze_input",
    "apply_profile",
    "build_manifest",
    "build_compilation_report",
    "evaluate",
    "extract",
    "extract_from_arrays",
    "extract_from_mesh",
    "extract_from_rigged_mesh",
    "get_profile",
    "read_phys",
    "resolve_config",
    "validate_phys",
    "validate_compilation_report",
    "verify_bundle",
    "write_manifest",
]
