"""Content hashing and toolchain identity, shared by both front doors.

The service layer used to own these (in ``chitin_service.store.Store``), so
bundles produced by the ``chitin`` CLI carried no provenance at all. They live
in core now: the CLI, the exporter, and the service all hash inputs, configs,
and outputs the same way, and the provenance ``manifest.json`` is built from
them.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy

# Dependencies whose version actually shapes the produced geometry. Pinned into
# the compiler identity so an upgrade of any of them invalidates output-hash
# reuse (see the cache-verifiability caveat in manifest.py).
SHAPING_DEPS = ("coacd", "open3d", "trimesh", "numpy")


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_config(config_dict: dict) -> str:
    """Stable SHA-256 over a config dict (key order independent)."""
    blob = json.dumps(config_dict, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def base_version() -> str:
    import chitin

    return chitin.__version__


def dependency_versions() -> dict[str, str]:
    """Installed versions of the shaping dependencies (absent ones omitted)."""
    versions: dict[str, str] = {}
    for dep in SHAPING_DEPS:
        try:
            versions[dep] = importlib.metadata.version(dep)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def compiler_version() -> str:
    """A single string tying the chitin version to its shaping deps, e.g.
    ``0.1.0+coacd1.0.5+trimesh4.4.1+numpy2.1.0``."""
    parts = [base_version()]
    for dep, ver in dependency_versions().items():
        parts.append(f"{dep}{ver}")
    return "+".join(parts)


# ---------------------------------------------------------------------------
# Logical build identity (lingot #64 / chitin #100)
# ---------------------------------------------------------------------------

# Semantic algorithm version, independent of the chitin package version.
# Bump MAJOR for output-incompatible algorithm changes (different geometry from
# same input), MINOR for output-compatible additions (new metrics, new config
# keys with backward-compatible defaults), PATCH for bug fixes that do not
# change output on conforming inputs.
# A docs-only chitin release does NOT bump this.
ALGORITHM_VERSION = "1.0.0"

# Numerical policy version: acceptance thresholds and profile definitions.
# Changes here invalidate verdicts, not geometry — they signal "re-evaluate
# acceptance" not "recompile".
NUMERICAL_POLICY_VERSION = "1.0.0"

# Config fields that affect output geometry. Excludes operational settings
# (timeout, deterministic-mode threading, log level) which affect runtime
# behavior but not the mathematical output.
_OUTPUT_AFFECTING_CONFIG_FIELDS = (
    "concavity",
    "opacity_threshold",
    "poisson_depth",
    "min_hull_vertices",
    "max_hulls",
    "opacity_is_logit",
    "coacd_preprocess_mode",
    "coacd_preprocess_resolution",
    "coacd_adaptive_preprocess",
    "max_decompose_vertices",
    "lod_concavities",
    "splat_scale_is_log",
    "splat_surface_ratio",
    "spatial_split_threshold",
    "poisson_density_quantile",
    "surface_proximity_filter",
    "thin_shell",
    "thin_shell_thickness",
    "flatness_threshold",
    "auto_environment",
    "force_environment",
    "seam_repair",
    "snug_fit",
    "target_height",
    "target_footprint",
    "up_axis",
    "flat_aspect_ratio",
)


def effective_input_digest(vertices: "numpy.ndarray", faces: "numpy.ndarray") -> str:
    """SHA-256 of the post-preprocessing canonical geometry entering CoACD.

    This is the effective input — not the source file hash. A remeshed mesh
    has a different effective input than the unmodified mesh even if the
    source file is the same.
    """
    v_bytes = vertices.astype("float32").tobytes()
    f_bytes = faces.astype("int32").tobytes()
    return f"sha256:{hashlib.sha256(v_bytes + f_bytes).hexdigest()}"


def output_affecting_config_digest(config) -> str:
    """SHA-256 of the resolved config fields that affect output geometry."""
    from dataclasses import asdict

    full = asdict(config)
    filtered = {k: full[k] for k in _OUTPUT_AFFECTING_CONFIG_FIELDS if k in full}
    blob = json.dumps(filtered, sort_keys=True, default=str).encode()
    return f"sha256:{hashlib.sha256(blob).hexdigest()}"


def build_logical_identity(
    vertices: "numpy.ndarray",
    faces: "numpy.ndarray",
    config,
) -> dict:
    """The logical build identity: what was asked for, independent of toolchain."""
    return {
        "effective_input_digest": effective_input_digest(vertices, faces),
        "algorithm_version": ALGORITHM_VERSION,
        "numerical_policy_version": NUMERICAL_POLICY_VERSION,
        "config_digest": output_affecting_config_digest(config),
    }


def build_realization(logical_identity: dict, artifact_sha256: str | None) -> dict:
    """Full realization record: logical identity + toolchain + artifact."""
    return {
        "logical_build_identity": logical_identity,
        "realization": {
            "runtime": {
                "kind": "python_native",
                "implementation": "chitin",
                "version": base_version(),
                "compiler_version": compiler_version(),
                "dependencies": dependency_versions(),
            },
        },
        "artifact_digest": f"sha256:{artifact_sha256}" if artifact_sha256 else None,
    }
