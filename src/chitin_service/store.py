# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import chitin

from .models import Job, JobConfig, JobEvent, JobInput, JobStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    input_uri TEXT NOT NULL,
    input_kind TEXT NOT NULL DEFAULT 'auto',
    original_filename TEXT,
    outputs TEXT NOT NULL DEFAULT '["phys","json"]',
    profile TEXT NOT NULL DEFAULT 'interactive',
    config_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created',
    input_hash TEXT,
    config_hash TEXT,
    compiler_version TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    timestamp TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT
);
CREATE INDEX IF NOT EXISTS idx_cache
    ON jobs (input_hash, config_hash, compiler_version, status);
CREATE INDEX IF NOT EXISTS idx_events_job
    ON job_events (job_id);
"""


class Store:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.db_path = data_dir / "chitin.db"
        self.artifacts_dir = data_dir / "artifacts"
        self.inputs_dir = data_dir / "inputs"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_job(self, job: Job) -> None:
        config_json = json.dumps(job.config.to_dict())
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO jobs
                   (id, input_uri, input_kind, original_filename, outputs,
                    profile, config_json, status, input_hash, config_hash,
                    compiler_version, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job.id,
                    job.input.uri,
                    job.input.kind,
                    job.input.original_filename,
                    json.dumps(job.outputs),
                    job.profile,
                    config_json,
                    job.status.value,
                    job.input_hash,
                    job.config_hash,
                    job.compiler_version,
                    job.created_at.isoformat(),
                ),
            )
            for ev in job.events:
                self._insert_event(conn, job.id, ev)

    def get_job(self, job_id: str) -> Job | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            events = conn.execute(
                "SELECT * FROM job_events WHERE job_id = ? ORDER BY id",
                (job_id,),
            ).fetchall()
        return self._row_to_job(row, events)

    def update_job(self, job: Job) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE jobs SET status=?, input_hash=?, config_hash=?, error=?
                   WHERE id=?""",
                (
                    job.status.value,
                    job.input_hash,
                    job.config_hash,
                    job.error,
                    job.id,
                ),
            )
            existing = conn.execute(
                "SELECT COUNT(*) FROM job_events WHERE job_id = ?", (job.id,)
            ).fetchone()[0]
            for ev in job.events[existing:]:
                self._insert_event(conn, job.id, ev)

    def find_cached(
        self, input_hash: str, config_hash: str, compiler_version: str
    ) -> Job | None:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT * FROM jobs
                   WHERE input_hash=? AND config_hash=? AND compiler_version=?
                   AND status='complete'
                   ORDER BY created_at DESC LIMIT 1""",
                (input_hash, config_hash, compiler_version),
            ).fetchone()
            if row is None:
                return None
            events = conn.execute(
                "SELECT * FROM job_events WHERE job_id = ? ORDER BY id",
                (row["id"],),
            ).fetchall()
        return self._row_to_job(row, events)

    def save_input(self, job_id: str, data: bytes, filename: str) -> Path:
        job_dir = self.inputs_dir / job_id
        job_dir.mkdir(exist_ok=True)
        safe_name = Path(filename).name or "input"
        path = job_dir / safe_name
        path.write_bytes(data)
        return path

    def get_input_path(self, job_id: str) -> Path | None:
        job_dir = self.inputs_dir / job_id
        if not job_dir.exists():
            return None
        files = list(job_dir.iterdir())
        return files[0] if files else None

    def job_artifact_dir(self, job_id: str) -> Path:
        d = self.artifacts_dir / job_id
        d.mkdir(exist_ok=True)
        return d

    def copy_artifacts(self, source_job_id: str, dest_job_id: str) -> None:
        src = self.artifacts_dir / source_job_id
        dst = self.artifacts_dir / dest_job_id
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)

    def _insert_event(
        self, conn: sqlite3.Connection, job_id: str, event: JobEvent
    ) -> None:
        conn.execute(
            "INSERT INTO job_events (job_id, timestamp, status, message) VALUES (?, ?, ?, ?)",
            (job_id, event.timestamp.isoformat(), event.status.value, event.message),
        )

    def _row_to_job(self, row: sqlite3.Row, event_rows: list) -> Job:
        cfg = json.loads(row["config_json"])
        events = [
            JobEvent(
                timestamp=datetime.fromisoformat(e["timestamp"]),
                status=JobStatus(e["status"]),
                message=e["message"],
            )
            for e in event_rows
        ]
        return Job(
            id=row["id"],
            input=JobInput(
                uri=row["input_uri"],
                kind=row["input_kind"],
                original_filename=row["original_filename"],
            ),
            outputs=json.loads(row["outputs"]),
            profile=row["profile"],
            config=JobConfig(
                concavity=cfg.get("concavity", 0.05),
                opacity_threshold=cfg.get("opacity_threshold", 0.5),
                poisson_depth=cfg.get("poisson_depth", 8),
                min_hull_vertices=cfg.get("min_hull_vertices", 4),
                max_hulls=cfg.get("max_hulls", 256),
                opacity_is_logit=cfg.get("opacity_is_logit", False),
                coacd_preprocess_mode=cfg.get("coacd_preprocess_mode", "auto"),
                coacd_preprocess_resolution=cfg.get("coacd_preprocess_resolution", 50),
                max_decompose_vertices=cfg.get("max_decompose_vertices", 200_000),
            ),
            status=JobStatus(row["status"]),
            input_hash=row["input_hash"],
            config_hash=row["config_hash"],
            compiler_version=row["compiler_version"],
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            events=events,
        )

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def hash_config(config: JobConfig, outputs: list[str]) -> str:
        d = config.to_dict()
        d["outputs"] = sorted(outputs)
        blob = json.dumps(d, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    @staticmethod
    def compiler_version() -> str:
        return chitin.__version__ if hasattr(chitin, "__version__") else "0.1.0"
