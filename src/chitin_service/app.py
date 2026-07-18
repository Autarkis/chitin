from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .models import Job, JobConfig, JobInput, JobStatus
from .store import Store
from .worker import run_job

app = FastAPI(title="chitin", version="0.1.0")

_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        data_dir = Path.home() / ".local" / "share" / "chitin"
        _store = Store(data_dir)
    return _store


def set_store(store: Store) -> None:
    global _store
    _store = store


VALID_OUTPUTS = {"json", "phys", "usd"}


class EventResponse(BaseModel):
    timestamp: str
    status: str
    message: str | None = None


class JobResponse(BaseModel):
    id: str
    status: str
    input_uri: str
    input_kind: str
    original_filename: str | None = None
    outputs: list[str]
    profile: str
    config: dict
    input_hash: str | None = None
    cached_from: str | None = None
    error: str | None = None
    created_at: str
    events: list[EventResponse]


def _job_response(job: Job, cached_from: str | None = None) -> JobResponse:
    return JobResponse(
        id=job.id,
        status=job.status.value,
        input_uri=job.input.uri,
        input_kind=job.input.kind,
        original_filename=job.input.original_filename,
        outputs=job.outputs,
        profile=job.profile,
        config=job.config.to_dict(),
        input_hash=job.input_hash,
        cached_from=cached_from,
        error=job.error,
        created_at=job.created_at.isoformat(),
        events=[
            EventResponse(
                timestamp=e.timestamp.isoformat(),
                status=e.status.value,
                message=e.message,
            )
            for e in job.events
        ],
    )


@app.post("/v1/jobs", response_model=JobResponse, status_code=201)
async def submit_job(
    file: UploadFile,
    outputs: str = "phys,json",
    profile: str = "interactive",
    concavity: float = 0.05,
    opacity_threshold: float = 0.5,
    poisson_depth: int | None = None,
    min_hull_vertices: int = 4,
    max_hulls: int = 256,
    opacity_is_logit: bool = False,
    coacd_preprocess_mode: str = "auto",
    coacd_preprocess_resolution: int = 50,
    max_decompose_vertices: int = 200_000,
):
    store = get_store()
    output_list = [f.strip() for f in outputs.split(",")]
    for fmt in output_list:
        if fmt not in VALID_OUTPUTS:
            raise HTTPException(400, f"unsupported output format: {fmt}")

    if coacd_preprocess_mode not in ("auto", "on", "off"):
        raise HTTPException(
            400, f"invalid coacd_preprocess_mode: {coacd_preprocess_mode}"
        )

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "empty file")

    filename = file.filename or "input"
    job_id = uuid.uuid4().hex[:12]
    job_config = JobConfig(
        concavity=concavity,
        opacity_threshold=opacity_threshold,
        poisson_depth=poisson_depth,
        min_hull_vertices=min_hull_vertices,
        max_hulls=max_hulls,
        opacity_is_logit=opacity_is_logit,
        coacd_preprocess_mode=coacd_preprocess_mode,
        coacd_preprocess_resolution=coacd_preprocess_resolution,
        max_decompose_vertices=max_decompose_vertices,
    )

    input_hash = Store.hash_bytes(data)
    config_hash = Store.hash_config(job_config, output_list, Path(filename).suffix)
    compiler_ver = Store.compiler_version()

    cached = store.find_cached(input_hash, config_hash, compiler_ver)
    if cached is not None:
        store.copy_artifacts(cached.id, job_id)
        job = Job(
            id=job_id,
            input=JobInput(
                uri=f"upload://{filename}", kind="auto", original_filename=filename
            ),
            outputs=output_list,
            profile=profile,
            config=job_config,
            status=JobStatus.CREATED,
            input_hash=input_hash,
            config_hash=config_hash,
            compiler_version=compiler_ver,
        )
        job.transition(JobStatus.UPLOADED)
        job.transition(JobStatus.PREFLIGHTED)
        job.transition(JobStatus.QUEUED)
        job.transition(JobStatus.RUNNING, "cache hit")
        job.transition(JobStatus.EXPORTING)
        job.transition(JobStatus.COMPLETE, f"cached from {cached.id}")
        store.create_job(job)
        return _job_response(job, cached_from=cached.id)

    job = Job(
        id=job_id,
        input=JobInput(
            uri=f"upload://{filename}", kind="auto", original_filename=filename
        ),
        outputs=output_list,
        profile=profile,
        config=job_config,
        input_hash=input_hash,
        config_hash=config_hash,
        compiler_version=compiler_ver,
    )
    store.create_job(job)

    input_path = store.save_input(job_id, data, filename)
    job.transition(JobStatus.UPLOADED)
    store.update_job(job)

    from chitin.preflight import check as preflight_check

    pf = preflight_check(input_path)
    if pf.level == "red":
        job.transition(JobStatus.REJECTED, pf.message)
        store.update_job(job)
        raise HTTPException(413, pf.message)

    job.transition(JobStatus.PREFLIGHTED, pf.level)
    store.update_job(job)

    job.transition(JobStatus.QUEUED)
    store.update_job(job)

    job = run_job(store, job)
    return _job_response(job)


@app.get("/v1/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    store = get_store()
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return _job_response(job)


@app.get("/v1/jobs/{job_id}/events")
async def get_events(job_id: str):
    store = get_store()
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return {
        "events": [
            {
                "timestamp": e.timestamp.isoformat(),
                "status": e.status.value,
                "message": e.message,
            }
            for e in job.events
        ]
    }


@app.get("/v1/jobs/{job_id}/artifacts")
async def list_artifacts(job_id: str):
    store = get_store()
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if not job.status.terminal:
        raise HTTPException(409, f"job is {job.status.value}")
    if job.status in (JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.REJECTED):
        raise HTTPException(409, f"job {job.status.value}, no artifacts")

    artifact_dir = store.artifacts_dir / job_id
    if not artifact_dir.exists():
        return {"artifacts": []}
    return {
        "artifacts": [f.name for f in sorted(artifact_dir.iterdir()) if f.is_file()]
    }


@app.get("/v1/jobs/{job_id}/artifacts/{filename}")
async def download_artifact(job_id: str, filename: str):
    store = get_store()
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status != JobStatus.COMPLETE:
        raise HTTPException(409, f"job is {job.status.value}")

    artifact_dir = store.job_artifact_dir(job_id)
    path = (artifact_dir / filename).resolve()
    if not path.is_relative_to(artifact_dir.resolve()) or not path.exists():
        raise HTTPException(404, f"artifact not found: {filename}")
    return FileResponse(path)


@app.post("/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    store = get_store()
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status.terminal:
        raise HTTPException(409, f"job already {job.status.value}")
    try:
        job.transition(JobStatus.CANCELLED, "cancelled by client")
    except ValueError:
        raise HTTPException(409, f"cannot cancel job in state {job.status.value}")
    store.update_job(job)
    return _job_response(job)
