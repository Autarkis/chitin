from __future__ import annotations

import json
import traceback

import chitin

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

        config = job.config.to_core_config()
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

        report = _build_report(result, config, job)
        (artifact_dir / "report.json").write_text(json.dumps(report, indent=2))

        job.transition(JobStatus.COMPLETE, f"{len(result.hulls)} hulls generated")
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
    result: chitin.ExtractionResult, config: chitin.Config, job: Job
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
        "status": "complete",
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
        "config": job.config.to_dict(),
        "compiler_version": job.compiler_version,
        "outputs": job.outputs,
        "artifacts": {
            fmt: ARTIFACT_NAMES[fmt] for fmt in job.outputs if fmt in ARTIFACT_NAMES
        },
    }
    return report
