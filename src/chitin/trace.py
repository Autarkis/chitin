"""Stage-level trace recorder for oracle replay.

Records the geometry state at each pipeline stage boundary. Events are
timestampless and carry input/output digests so two traces can be compared
to isolate the first divergent stage.

Instrumentation boundary
------------------------
CoACD runs as an opaque subprocess (_coacd_worker.py) via its Python bindings
(coacd.run_coacd). Internal state — candidate splitting planes, MCTS
transitions, per-face predicate classifications, clip intersections — is not
observable without patching the C++ source. This tracer operates at the Python
stage boundary: it records geometry entering and leaving each stage, and hull
geometry produced by decomposition. When two traces diverge, the replay diff
reports the first stage whose output differs and, for decomposition stages,
a semantic hull-level comparison (hull count, per-hull vertex/face counts,
max vertex displacement, winding consistency). Predicate-level divergence
isolation requires a trace-instrumented CoACD build (future work, tracked
as chitin #91 Phase 2).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

TRACE_SCHEMA_VERSION = "1.0.0"


def _geometry_digest(vertices: np.ndarray, faces: np.ndarray | None = None) -> str:
    """SHA-256 of canonical geometry arrays."""
    h = hashlib.sha256()
    h.update(vertices.astype(np.float32).tobytes())
    if faces is not None:
        h.update(faces.astype(np.int32).tobytes())
    return f"sha256:{h.hexdigest()}"


def _hull_digest(hulls: list[tuple[np.ndarray, np.ndarray]]) -> str:
    """SHA-256 over an ordered list of (vertices, indices) hull pairs."""
    h = hashlib.sha256()
    h.update(len(hulls).to_bytes(4, "little"))
    for verts, indices in hulls:
        v = np.asarray(verts, dtype=np.float32)
        idx = np.asarray(indices, dtype=np.uint32)
        h.update(v.tobytes())
        h.update(idx.tobytes())
    return f"sha256:{h.hexdigest()}"


@dataclass
class StageEvent:
    """One pipeline stage boundary observation."""

    stage: str
    input_digest: str
    output_digest: str
    input_shape: list[int]
    output_shape: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


class TraceRecorder:
    """Captures geometry state at each pipeline stage boundary.

    No timestamps — traces are pure functions of input geometry + config.
    """

    def __init__(self, config_dict: dict | None = None) -> None:
        self.events: list[StageEvent] = []
        self.config_dict = config_dict or {}
        self._blobs: dict[str, np.ndarray] = {}

    def record_stage(
        self,
        stage: str,
        input_vertices: np.ndarray,
        output_vertices: np.ndarray,
        input_faces: np.ndarray | None = None,
        output_faces: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        in_digest = _geometry_digest(input_vertices, input_faces)
        out_digest = _geometry_digest(output_vertices, output_faces)
        event = StageEvent(
            stage=stage,
            input_digest=in_digest,
            output_digest=out_digest,
            input_shape=list(input_vertices.shape),
            output_shape=list(output_vertices.shape),
            metadata=metadata or {},
        )
        self.events.append(event)
        blob_key = f"{len(self.events) - 1}_{stage}"
        self._blobs[f"{blob_key}_in_v"] = input_vertices.astype(np.float32)
        self._blobs[f"{blob_key}_out_v"] = output_vertices.astype(np.float32)
        if input_faces is not None:
            self._blobs[f"{blob_key}_in_f"] = input_faces.astype(np.int32)
        if output_faces is not None:
            self._blobs[f"{blob_key}_out_f"] = output_faces.astype(np.int32)

    def record_decompose(
        self,
        stage: str,
        input_vertices: np.ndarray,
        input_faces: np.ndarray,
        output_hulls: list[tuple[np.ndarray, np.ndarray]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a decomposition stage (mesh in, hull list out)."""
        in_digest = _geometry_digest(input_vertices, input_faces)
        out_digest = _hull_digest(output_hulls)
        total_verts = sum(len(v) for v, _ in output_hulls)
        event = StageEvent(
            stage=stage,
            input_digest=in_digest,
            output_digest=out_digest,
            input_shape=list(input_vertices.shape),
            output_shape=[len(output_hulls), total_verts],
            metadata={**(metadata or {}), "hull_count": len(output_hulls)},
        )
        self.events.append(event)
        blob_key = f"{len(self.events) - 1}_{stage}"
        self._blobs[f"{blob_key}_in_v"] = input_vertices.astype(np.float32)
        self._blobs[f"{blob_key}_in_f"] = input_faces.astype(np.int32)
        for i, (v, idx) in enumerate(output_hulls):
            self._blobs[f"{blob_key}_hull_{i}_v"] = np.asarray(v, dtype=np.float32)
            self._blobs[f"{blob_key}_hull_{i}_i"] = np.asarray(idx, dtype=np.uint32)

    def to_dict(self) -> dict:
        return {
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "config": self.config_dict,
            "events": [asdict(e) for e in self.events],
        }

    def save(self, directory: str | Path) -> Path:
        """Write trace index JSON + geometry blobs to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        index_path = directory / "trace.json"
        index_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if self._blobs:
            blobs_path = directory / "geometry.npz"
            np.savez_compressed(blobs_path, **self._blobs)
        return index_path

    @classmethod
    def load(cls, directory: str | Path) -> TraceRecorder:
        """Load a saved trace from directory."""
        directory = Path(directory)
        index_path = directory / "trace.json"
        data = json.loads(index_path.read_text(encoding="utf-8"))
        recorder = cls(config_dict=data.get("config", {}))
        for event_dict in data.get("events", []):
            recorder.events.append(StageEvent(**event_dict))
        blobs_path = directory / "geometry.npz"
        if blobs_path.exists():
            with np.load(blobs_path) as npz:
                for key in npz.files:
                    recorder._blobs[key] = npz[key]
        return recorder
