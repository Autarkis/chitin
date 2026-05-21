# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone


class JobStatus(str, enum.Enum):
    CREATED = "created"
    UPLOADED = "uploaded"
    PREFLIGHTED = "preflighted"
    QUEUED = "queued"
    RUNNING = "running"
    EXPORTING = "exporting"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

    @property
    def terminal(self) -> bool:
        return self in (
            JobStatus.COMPLETE,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.REJECTED,
        )


VALID_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.CREATED: {JobStatus.UPLOADED, JobStatus.FAILED},
    JobStatus.UPLOADED: {JobStatus.PREFLIGHTED, JobStatus.REJECTED, JobStatus.FAILED},
    JobStatus.PREFLIGHTED: {JobStatus.QUEUED, JobStatus.REJECTED},
    JobStatus.QUEUED: {JobStatus.RUNNING, JobStatus.CANCELLED},
    JobStatus.RUNNING: {JobStatus.EXPORTING, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.EXPORTING: {JobStatus.COMPLETE, JobStatus.FAILED},
    JobStatus.COMPLETE: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
    JobStatus.REJECTED: set(),
}


@dataclass
class JobEvent:
    timestamp: datetime
    status: JobStatus
    message: str | None = None


@dataclass
class JobInput:
    uri: str
    kind: str = "auto"
    original_filename: str | None = None


@dataclass
class JobConfig:
    concavity: float = 0.05
    opacity_threshold: float = 0.5
    poisson_depth: int | None = None
    min_hull_vertices: int = 4
    max_hulls: int = 256
    opacity_is_logit: bool = False
    coacd_preprocess_mode: str = "auto"
    coacd_preprocess_resolution: int = 50
    max_decompose_vertices: int = 200_000

    def to_core_config(self):
        import chitin

        return chitin.Config(
            concavity=self.concavity,
            opacity_threshold=self.opacity_threshold,
            poisson_depth=self.poisson_depth,
            min_hull_vertices=self.min_hull_vertices,
            max_hulls=self.max_hulls,
            opacity_is_logit=self.opacity_is_logit,
            coacd_preprocess_mode=self.coacd_preprocess_mode,
            coacd_preprocess_resolution=self.coacd_preprocess_resolution,
            max_decompose_vertices=self.max_decompose_vertices,
        )

    def to_dict(self) -> dict:
        return {
            "concavity": self.concavity,
            "opacity_threshold": self.opacity_threshold,
            "poisson_depth": self.poisson_depth,
            "min_hull_vertices": self.min_hull_vertices,
            "max_hulls": self.max_hulls,
            "opacity_is_logit": self.opacity_is_logit,
            "coacd_preprocess_mode": self.coacd_preprocess_mode,
            "coacd_preprocess_resolution": self.coacd_preprocess_resolution,
            "max_decompose_vertices": self.max_decompose_vertices,
        }


@dataclass
class Job:
    id: str
    input: JobInput
    outputs: list[str] = field(default_factory=lambda: ["phys", "json"])
    profile: str = "interactive"
    config: JobConfig = field(default_factory=JobConfig)
    status: JobStatus = JobStatus.CREATED
    input_hash: str | None = None
    config_hash: str | None = None
    compiler_version: str = "0.1.0"
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    events: list[JobEvent] = field(default_factory=list)

    def transition(self, new_status: JobStatus, message: str | None = None) -> None:
        allowed = VALID_TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            raise ValueError(
                f"invalid transition: {self.status.value} -> {new_status.value}"
            )
        self.status = new_status
        self.events.append(
            JobEvent(
                timestamp=datetime.now(timezone.utc),
                status=new_status,
                message=message,
            )
        )
