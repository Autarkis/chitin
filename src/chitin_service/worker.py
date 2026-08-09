from __future__ import annotations

import dataclasses
import json
import traceback
from pathlib import Path

import chitin
from chitin import provenance
from chitin.acceptance import (
    Verdict,
    apply_profile,
    evaluate,
    get_profile,
    report_metrics,
)
from chitin.manifest import MANIFEST_FILENAME, write_manifest
from chitin.report import build_compilation_report, select_primary_artifact

from .models import Job, JobStatus
from .store import Store

ARTIFACT_NAMES = {
    "json": "colliders.json",
    "phys": "colliders.phys",
    "usd": "colliders.usda",
}


def run_job(store: Store, job: Job) -> Job:
    try:
        job.transition(JobStatus.RUNNING)
        store.update_job(job)

        input_path = store.get_input_path(job.id)
        if input_path is None:
            raise FileNotFoundError(f"no input file for job {job.id}")

        # The profile is no longer inert: it presets the config (where the
        # client left defaults) and supplies the acceptance policy.
        profile = get_profile(job.profile)
        config = apply_profile(job.config.to_core_config(), profile)
        result = chitin.extract(input_path, config=config)

        job.transition(JobStatus.EXPORTING)
        store.update_job(job)

        artifact_dir = store.job_artifact_dir(job.id)
        for fmt in job.outputs:
            if fmt == "json":
                result.to_json(artifact_dir / "colliders.json")
            elif fmt == "phys":
                result.to_phys(artifact_dir / "colliders.phys")
            elif fmt == "usd":
                result.to_usd(artifact_dir / "colliders.usda")

        verdict = evaluate(profile.policy, report_metrics(result))
        report = _build_report(result, config, job, verdict, artifact_dir)
        (artifact_dir / "report.json").write_text(json.dumps(report, indent=2))

        # Provenance manifest over every artifact written above (report.json
        # included), so a service bundle is auditable just like a CLI one.
        output_files = [
            p.name
            for p in sorted(artifact_dir.iterdir())
            if p.is_file() and p.name != MANIFEST_FILENAME
        ]
        resolved_dict = (
            result.resolved.to_dict()
            if result.resolved is not None and hasattr(result.resolved, "to_dict")
            else None
        )
        write_manifest(
            artifact_dir,
            output_files=output_files,
            input_path=input_path,
            config_dict=dataclasses.asdict(config),
            resolved_dict=resolved_dict,
            metrics=report_metrics(result),
            warnings=report["warnings"],
            verdict=verdict.to_dict(),
            compilation_report=report["compilation_report"],
        )

        if verdict.passed:
            job.transition(JobStatus.COMPLETE, f"{len(result.hulls)} hulls generated")
        else:
            job.transition(
                JobStatus.REJECTED,
                "; ".join(verdict.reasons) or "failed acceptance",
            )
        store.update_job(job)

    except Exception as exc:
        tb = traceback.format_exc()
        job.error = f"{type(exc).__name__}: {exc}"

        artifact_dir = store.job_artifact_dir(job.id)
        (artifact_dir / "logs.txt").write_text(tb)

        if job.status == JobStatus.RUNNING:
            job.transition(JobStatus.FAILED, str(exc))
        elif job.status == JobStatus.EXPORTING:
            job.transition(JobStatus.FAILED, f"export failed: {exc}")
        store.update_job(job)

    return job


def _build_report(
    result: chitin.ExtractionResult,
    config: chitin.Config,
    job: Job,
    verdict: Verdict,
    artifact_dir: Path,
) -> dict:
    plan = result.build_plan
    warnings = []

    if plan and plan.decimated:
        warnings.append("mesh was decimated before decomposition")

    if plan and plan.detected.get("decimation_skipped"):
        n = plan.detected["decimation_skipped"]
        warnings.append(
            f"mesh has {n} vertices over max_decompose_vertices but decimation was "
            "skipped (Open3D not installed); install chitin[splat] to enable it"
        )

    if plan and plan.detected.get("fallback_hulls"):
        n = plan.detected["fallback_hulls"]
        warnings.append(f"{n} AABB fallback hull(s) substituted after a CoACD timeout")

    bones_with_colliders = 0
    if result.bones:
        bone_names_with_hulls = {h.bone_name for h in result.hulls if h.bone_name}
        bones_with_colliders = len(bone_names_with_hulls)
        bones_skipped = plan.detected.get("bones_skipped", 0) if plan else 0
        if bones_skipped > 0:
            warnings.append(
                f"{bones_skipped} bones had too little geometry for hull generation"
            )

    report = {
        "status": "complete" if verdict.passed else "rejected",
        "profile": job.profile,
        "verdict": verdict.to_dict(),
        "input_kind": plan.input_kind if plan else "unknown",
        "collider_kind": plan.collider_kind if plan else "unknown",
        "pipeline": plan.pipeline if plan else [],
        "hull_count": len(result.hulls),
        "source_vertices": result.source_vertex_count,
        "processed_vertices": plan.processed_vertices
        if plan
        else result.mesh_vertex_count,
        "mesh_vertices": result.mesh_vertex_count,
        "rigged": result.bones is not None,
        "bones_with_colliders": bones_with_colliders,
        "bones_total": len(result.bones) if result.bones else 0,
        "warnings": warnings,
        "detected": plan.detected if plan else {},
        # Both, because they differ: the profile presets fields the request
        # left at their defaults, so `config` is what was asked for and
        # `effective_config` is what the geometry and the manifest were
        # actually built from.
        "config": job.config.to_dict(),
        "effective_config": dataclasses.asdict(config),
        "compiler_version": job.compiler_version,
        "outputs": job.outputs,
        "artifacts": {
            fmt: ARTIFACT_NAMES[fmt] for fmt in job.outputs if fmt in ARTIFACT_NAMES
        },
    }
    artifacts = report["artifacts"]
    primary_path = select_primary_artifact(
        artifact_dir / name for name in artifacts.values()
    )
    artifact_bytes = (
        primary_path.stat().st_size
        if primary_path is not None and primary_path.is_file()
        else None
    )
    artifact_sha256 = (
        provenance.hash_file(primary_path)
        if primary_path is not None and primary_path.is_file()
        else None
    )
    report["compilation_report"] = build_compilation_report(
        result,
        profile=job.profile,
        verdict=verdict,
        warnings=warnings,
        requested_config=job.config.to_dict(),
        effective_config=dataclasses.asdict(config),
        artifacts=artifacts,
        artifact_bytes=artifact_bytes,
        artifact_sha256=artifact_sha256,
    ).to_dict()
    return report
