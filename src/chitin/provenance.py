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

    return getattr(chitin, "__version__", "0.1.0")


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
