"""Provenance ``manifest.json`` for a compiled bundle.

A bundle ships ``colliders.{phys,json,usda}`` plus ``report.json``, but nothing
canonically tied input, output, config, and toolchain together — and the CLI
door produced bundles with no provenance at all. The manifest is that record,
emitted from core so both front doors (CLI exporter and service) carry the same
guarantee.

Two promises live here:

* **Integrity / tamper-evident** — the manifest declares the SHA-256 of every
  emitted file, so :func:`verify_bundle` can confirm a bundle matches what it
  claims. Always available.
* **Cache-verifiability** — deterministic builds are reproducible within the
  runtime/toolchain identified by the report. Native Python and browser WASM
  are separate reproducibility domains unless a cross-runtime conformance test
  explicitly proves otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

from chitin import provenance
from chitin.phys import WRITE_VERSION as PHYS_WRITE_VERSION

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "manifest.json"


def quality_warnings(result) -> list[str]:
    """Non-fatal quality flags derived from the build plan.

    A basic set for the exporter path; the service composes its own richer
    warnings for ``report.json`` and passes them straight through.
    """
    plan = result.build_plan
    if plan is None:
        return []
    detected = plan.detected
    warnings: list[str] = []
    if detected.get("fallback_hulls"):
        warnings.append(
            f"{detected['fallback_hulls']} AABB fallback hull(s) from CoACD timeout"
        )
    if plan.decimated:
        warnings.append("mesh was decimated before decomposition")
    if detected.get("decimation_skipped"):
        warnings.append(
            "mesh exceeded max_decompose_vertices but decimation was skipped "
            "(Open3D not installed)"
        )
    if detected.get("bones_skipped"):
        warnings.append(
            f"{detected['bones_skipped']} bones had too little geometry for hulls"
        )
    return warnings


def build_manifest(
    *,
    output_dir: str | Path,
    output_files: list[str],
    input_path: str | Path | None = None,
    config_dict: dict | None = None,
    resolved_dict: dict | None = None,
    metrics: dict | None = None,
    warnings: list[str] | None = None,
    verdict: dict | None = None,
    compilation_report: dict | None = None,
) -> dict:
    """Assemble the manifest dict.

    ``output_files`` are bundle-relative names; each existing one is hashed.
    ``verdict`` is an acceptance :meth:`~chitin.acceptance.Verdict.to_dict`
    (``None`` before acceptance lands, leaving a plain quality summary).
    """
    output_dir = Path(output_dir)

    outputs = []
    for name in output_files:
        path = output_dir / name
        if path.exists():
            outputs.append(
                {
                    "file": name,
                    "sha256": provenance.hash_file(path),
                    "bytes": path.stat().st_size,
                }
            )

    manifest: dict = {
        "manifest_version": MANIFEST_VERSION,
        "phys_version": PHYS_WRITE_VERSION,
        "compiler_version": provenance.compiler_version(),
        "dependency_versions": provenance.dependency_versions(),
        "outputs": outputs,
    }

    if input_path is not None:
        input_path = Path(input_path)
        manifest["input"] = {
            "filename": input_path.name,
            "sha256": provenance.hash_file(input_path),
            "bytes": input_path.stat().st_size,
        }

    if config_dict is not None:
        manifest["config"] = {
            "hash": provenance.hash_config(config_dict),
            "values": config_dict,
        }

    if resolved_dict is not None:
        manifest["resolved_config"] = resolved_dict

    quality: dict = {}
    if metrics is not None:
        quality["metrics"] = metrics
    if warnings:
        quality["warnings"] = list(warnings)
    if verdict is not None:
        quality["verdict"] = verdict
    if compilation_report is not None:
        quality["report"] = compilation_report
    if quality:
        manifest["quality"] = quality

    return manifest


def write_manifest(output_dir: str | Path, **kwargs) -> Path:
    """Build and write ``manifest.json`` into ``output_dir``. Returns its path."""
    output_dir = Path(output_dir)
    manifest = build_manifest(output_dir=output_dir, **kwargs)
    path = output_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def verify_bundle(bundle_dir: str | Path) -> list[str]:
    """Recompute declared output hashes against the files on disk.

    Returns a list of human-readable problems; an empty list means every
    declared artifact is present and matches its hash.
    """
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return [f"missing {MANIFEST_FILENAME}"]

    manifest = json.loads(manifest_path.read_text())
    problems: list[str] = []
    for entry in manifest.get("outputs", []):
        name = entry["file"]
        declared = entry["sha256"]
        path = bundle_dir / name
        if not path.exists():
            problems.append(f"{name}: declared in manifest but missing")
            continue
        actual = provenance.hash_file(path)
        if actual != declared:
            problems.append(
                f"{name}: hash mismatch "
                f"(declared {declared[:12]}…, actual {actual[:12]}…)"
            )
    return problems
