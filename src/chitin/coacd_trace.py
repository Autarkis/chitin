"""CoACD internal trace capture and replay (#108).

Parses JSONL + .npy trace output from the traced CoACD DLL into structured
Python dataclasses for use by the f32 predicate gate.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class TracedPlane:
    """A candidate cutting plane recorded by CoACD."""

    a: float
    b: float
    c: float
    d: float
    method: str  # "axis_aligned" or "ternary_refine"
    index: int

    @property
    def normal(self) -> np.ndarray:
        n = np.array([self.a, self.b, self.c], dtype=np.float64)
        length = np.linalg.norm(n)
        return n / length if length > 0 else n

    @property
    def offset(self) -> float:
        return self.d


@dataclass
class TracedClip:
    """Result of a single clip operation."""

    component_id: int
    plane: TracedPlane
    pos_verts: int
    pos_faces: int
    neg_verts: int
    neg_faces: int
    intersection_count: int
    # Loaded lazily from .npy if available
    pos_vertices: np.ndarray | None = None
    pos_triangles: np.ndarray | None = None
    neg_vertices: np.ndarray | None = None
    neg_triangles: np.ndarray | None = None
    # Oracle data (trace version 2+ only)
    input_vertices: np.ndarray | None = None  # (V, 3) float64 — mesh at Clip() entry
    input_faces: np.ndarray | None = None  # (F, 3) int32
    oracle_sides: np.ndarray | None = None  # (V,) int8 — C++ Side decisions
    cut_edges: np.ndarray | None = None  # (K, 2) int32 — split edge pairs
    cut_points: np.ndarray | None = None  # (K, 3) float64 — intersection points


@dataclass
class TracedCost:
    """Cost metric for a decomposition step."""

    component_id: int
    rv: float
    hb: float
    energy: float


@dataclass
class TracedMCTS:
    """MCTS transition record."""

    iteration: int
    reward: float
    worst_part: int
    visits: int


@dataclass
class ComponentState:
    """Mesh and convex hull state of a component at a given iteration."""

    iteration: int
    component_index: int
    mesh_vertices: np.ndarray  # (N, 3) float64
    mesh_faces: np.ndarray  # (M, 3) int32
    hull_vertices: np.ndarray  # (K, 3) float64
    hull_faces: np.ndarray  # (L, 3) int32


@dataclass
class CoACDTrace:
    """Complete trace of a single CoACD decomposition run."""

    call_id: int
    input_vertices: np.ndarray  # (N, 3) float64
    input_faces: np.ndarray  # (M, 3) int32
    planes: list[TracedPlane] = field(default_factory=list)
    clips: list[TracedClip] = field(default_factory=list)
    costs: list[TracedCost] = field(default_factory=list)
    mcts_transitions: list[TracedMCTS] = field(default_factory=list)
    component_states: list[ComponentState] = field(default_factory=list)
    output_parts: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)


def _load_npy(trace_dir: Path, filename: str) -> np.ndarray:
    """Load a .npy file from the trace directory."""
    return np.load(trace_dir / filename)


def load_trace(trace_dir: str | Path) -> CoACDTrace:
    """Load a complete CoACD trace from a directory.

    Expects exactly one .jsonl file (or specify call_id).
    """
    trace_dir = Path(trace_dir)
    jsonl_files = sorted(trace_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files in {trace_dir}")
    return _load_trace_file(trace_dir, jsonl_files[0])


def load_traces(trace_dir: str | Path) -> list[CoACDTrace]:
    """Load all traces from a directory (one per .jsonl file)."""
    trace_dir = Path(trace_dir)
    jsonl_files = sorted(trace_dir.glob("*.jsonl"))
    return [_load_trace_file(trace_dir, f) for f in jsonl_files]


def _load_array(
    trace_dir: Path, filename: str, npz: np.lib.npyio.NpzFile | None
) -> np.ndarray:
    """Load array from npz archive if available, else from individual .npy."""
    if npz is not None:
        key = filename.replace(".npy", "")
        if key in npz:
            return npz[key]
    return np.load(trace_dir / filename)


def _try_load_array(
    trace_dir: Path, filename: str, npz: np.lib.npyio.NpzFile | None
) -> np.ndarray | None:
    """Like _load_array but returns None if not found."""
    if not filename or not isinstance(filename, str):
        return None
    if npz is not None:
        key = filename.replace(".npy", "")
        if key in npz:
            return npz[key]
        return None
    npy_path = trace_dir / filename
    if npy_path.exists():
        return np.load(npy_path)
    return None


class _StreamCache:
    """Pre-loads concatenated stream arrays from npz once, slices per-clip cheaply."""

    def __init__(self, npz) -> None:
        self._data: dict[str, np.ndarray] = {}
        self._offsets: dict[str, np.ndarray] = {}
        if npz is None:
            return
        for key in list(npz.keys()):
            if key.endswith("_offsets"):
                base = key[: -len("_offsets")]
                self._offsets[base] = npz[key]
            elif key + "_offsets" in npz:
                self._data[key] = npz[key]

    def slice(self, name: str, idx: int) -> np.ndarray | None:
        if name not in self._data or name not in self._offsets:
            return None
        offsets = self._offsets[name]
        if idx + 1 >= len(offsets):
            return None
        start = int(offsets[idx])
        end = int(offsets[idx + 1])
        if start == end:
            return None
        return self._data[name][start:end]


def _is_stream_format(npz) -> bool:
    """Detect concatenated-stream npz (v3) vs per-entry npz (v2)."""
    if npz is None:
        return False
    if "clip_pos_verts" in npz and "clip_pos_verts_offsets" in npz:
        return True
    if "state_mesh_verts" in npz and "state_mesh_verts_offsets" in npz:
        return True
    return False


def _load_trace_file(trace_dir: Path, jsonl_path: Path) -> CoACDTrace:
    """Parse one .jsonl file and its associated .npy blobs."""
    events: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    if not events or events[0]["event"] != "begin":
        raise ValueError(f"Trace file {jsonl_path} does not start with 'begin' event")

    npz_files = sorted(trace_dir.glob("*_arrays.npz"))
    npz = np.load(npz_files[0]) if npz_files else None
    stream = _is_stream_format(npz)
    streams = _StreamCache(npz) if stream else None

    begin = events[0]
    call_id = begin["call_id"]

    if stream:
        input_verts = npz["input_verts"]
        input_faces = npz["input_faces"]
    elif "input_verts" in begin:
        input_verts = _load_array(trace_dir, begin["input_verts"], npz)
        input_faces = _load_array(trace_dir, begin["input_faces"], npz)
    else:
        input_verts = np.zeros((0, 3), dtype=np.float64)
        input_faces = np.zeros((0, 3), dtype=np.int32)

    trace = CoACDTrace(
        call_id=call_id,
        input_vertices=input_verts,
        input_faces=input_faces,
    )

    clip_idx = 0
    state_idx = 0

    for ev in events[1:]:
        kind = ev["event"]
        if kind == "plane":
            trace.planes.append(
                TracedPlane(
                    a=ev["a"],
                    b=ev["b"],
                    c=ev["c"],
                    d=ev["d"],
                    method=ev["method"],
                    index=ev["index"],
                )
            )
        elif kind == "clip":
            clip = TracedClip(
                component_id=ev["component_id"],
                plane=TracedPlane(
                    a=ev["a"],
                    b=ev["b"],
                    c=ev["c"],
                    d=ev["d"],
                    method="clip",
                    index=-1,
                ),
                pos_verts=ev["pos_num_vertices"],
                pos_faces=ev["pos_num_faces"],
                neg_verts=ev["neg_num_vertices"],
                neg_faces=ev["neg_num_faces"],
                intersection_count=ev["intersection_count"],
            )
            if stream:
                clip.pos_vertices = streams.slice("clip_pos_verts", clip_idx)
                clip.pos_triangles = streams.slice("clip_pos_faces", clip_idx)
                clip.neg_vertices = streams.slice("clip_neg_verts", clip_idx)
                clip.neg_triangles = streams.slice("clip_neg_faces", clip_idx)
                clip.input_vertices = streams.slice("clip_input_verts", clip_idx)
                clip.input_faces = streams.slice("clip_input_faces", clip_idx)
                clip.oracle_sides = streams.slice("clip_oracle_sides", clip_idx)
                clip.cut_points = streams.slice("clip_cut_points", clip_idx)
            else:
                for key, attr in [
                    ("pos_verts", "pos_vertices"),
                    ("pos_faces", "pos_triangles"),
                    ("neg_verts", "neg_vertices"),
                    ("neg_faces", "neg_triangles"),
                    ("input_verts", "input_vertices"),
                    ("input_faces", "input_faces"),
                    ("oracle_sides", "oracle_sides"),
                    ("cut_edges", "cut_edges"),
                    ("cut_points", "cut_points"),
                ]:
                    arr = _try_load_array(trace_dir, ev.get(key, ""), npz)
                    if arr is not None:
                        setattr(clip, attr, arr)
            clip_idx += 1
            trace.clips.append(clip)
        elif kind == "cost":
            trace.costs.append(
                TracedCost(
                    component_id=ev["component_id"],
                    rv=ev["rv"],
                    hb=ev["hb"],
                    energy=ev["energy"],
                )
            )
        elif kind == "mcts":
            trace.mcts_transitions.append(
                TracedMCTS(
                    iteration=ev["iteration"],
                    reward=ev["reward"],
                    worst_part=ev["worst_part"],
                    visits=ev["visits"],
                )
            )
        elif kind == "component_state":
            if stream:

                def _or_empty(arr, dtype=np.float64):
                    return arr if arr is not None else np.zeros((0, 3), dtype=dtype)

                state = ComponentState(
                    iteration=ev["iteration"],
                    component_index=ev["component_index"],
                    mesh_vertices=_or_empty(
                        streams.slice("state_mesh_verts", state_idx)
                    ),
                    mesh_faces=_or_empty(
                        streams.slice("state_mesh_faces", state_idx), np.int32
                    ),
                    hull_vertices=_or_empty(
                        streams.slice("state_hull_verts", state_idx)
                    ),
                    hull_faces=_or_empty(
                        streams.slice("state_hull_faces", state_idx), np.int32
                    ),
                )
            else:
                state = ComponentState(
                    iteration=ev["iteration"],
                    component_index=ev["component_index"],
                    mesh_vertices=_load_array(trace_dir, ev["mesh_verts"], npz),
                    mesh_faces=_load_array(trace_dir, ev["mesh_faces"], npz),
                    hull_vertices=_load_array(trace_dir, ev["hull_verts"], npz),
                    hull_faces=_load_array(trace_dir, ev["hull_faces"], npz),
                )
            state_idx += 1
            trace.component_states.append(state)
        elif kind == "end":
            if stream:
                num_parts = ev.get("num_parts", 0)
                for i in range(num_parts):
                    vk = f"output_{i}_verts"
                    fk = f"output_{i}_faces"
                    if npz is not None and vk in npz and fk in npz:
                        trace.output_parts.append((npz[vk], npz[fk]))
            elif "part_files" in ev:
                part_files = ev["part_files"]
                for i in range(0, len(part_files), 2):
                    verts = _load_array(trace_dir, part_files[i], npz)
                    faces = _load_array(trace_dir, part_files[i + 1], npz)
                    trace.output_parts.append((verts, faces))

    return trace


def capture_trace(
    vertices: np.ndarray,
    faces: np.ndarray,
    threshold: float = 0.05,
    preprocess_mode: str = "auto",
    preprocess_resolution: int = 50,
    max_convex_hull: int = -1,
    trace_dir: str | Path | None = None,
) -> CoACDTrace:
    """Run CoACD with tracing enabled, return parsed trace.

    Uses the system-installed coacd package (which must have the traced
    DLL deployed). Sets COACD_TRACE_DIR to a temp directory, runs the
    decomposition, and parses the output.
    """
    import coacd

    if trace_dir is None:
        trace_dir = Path(tempfile.mkdtemp(prefix="coacd_trace_"))
    else:
        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)

    os.environ["COACD_TRACE_DIR"] = str(trace_dir)
    try:
        mesh = coacd.Mesh(
            vertices.astype(np.float64),
            faces.astype(np.int32),
        )
        coacd.set_log_level("error")
        coacd.run_coacd(
            mesh,
            threshold=threshold,
            preprocess_mode=preprocess_mode,
            preprocess_resolution=preprocess_resolution,
            max_convex_hull=max_convex_hull,
        )
    finally:
        os.environ.pop("COACD_TRACE_DIR", None)

    return load_trace(trace_dir)


def _build_stream(
    clips: list[TracedClip],
    getter,
    dtype,
    cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate per-clip arrays into one stream + offset table."""
    parts = []
    offsets = [0]
    for c in clips:
        arr = getter(c)
        if arr is not None:
            if arr.ndim == 1 and cols > 1:
                arr = arr.reshape(-1, cols)
            parts.append(arr)
            offsets.append(offsets[-1] + arr.shape[0])
        else:
            offsets.append(offsets[-1])
    if parts:
        data = np.concatenate(parts).astype(dtype)
    else:
        data = np.zeros((0, cols), dtype=dtype)
    return data, np.array(offsets, dtype=np.int64)


def save_trace(trace: CoACDTrace, out_dir: str | Path) -> None:
    """Serialize a trace to disk for CI replay (no traced DLL needed).

    Uses concatenated-stream format: each clip field becomes two npz entries
    (data + offsets) instead of one entry per clip.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    arrays["input_verts"] = trace.input_vertices
    arrays["input_faces"] = trace.input_faces

    meta: dict = {
        "call_id": trace.call_id,
        "num_planes": len(trace.planes),
        "num_clips": len(trace.clips),
        "num_costs": len(trace.costs),
        "num_mcts": len(trace.mcts_transitions),
        "num_states": len(trace.component_states),
        "num_parts": len(trace.output_parts),
        "format": "stream_v3",
    }

    planes_data = [
        {"a": p.a, "b": p.b, "c": p.c, "d": p.d, "method": p.method, "index": p.index}
        for p in trace.planes
    ]
    with open(out_dir / "planes.json", "w") as f:
        json.dump(planes_data, f)

    clips_meta = []
    for c in trace.clips:
        clips_meta.append(
            {
                "component_id": c.component_id,
                "plane": {
                    "a": c.plane.a,
                    "b": c.plane.b,
                    "c": c.plane.c,
                    "d": c.plane.d,
                },
                "pos_verts": c.pos_verts,
                "pos_faces": c.pos_faces,
                "neg_verts": c.neg_verts,
                "neg_faces": c.neg_faces,
                "intersection_count": c.intersection_count,
            }
        )
    with open(out_dir / "clips.json", "w") as f:
        json.dump(clips_meta, f)

    stream_defs = [
        ("clip_pos_verts", lambda c: c.pos_vertices, np.float64, 3),
        ("clip_pos_faces", lambda c: c.pos_triangles, np.int32, 3),
        ("clip_neg_verts", lambda c: c.neg_vertices, np.float64, 3),
        ("clip_neg_faces", lambda c: c.neg_triangles, np.int32, 3),
        ("clip_input_verts", lambda c: c.input_vertices, np.float64, 3),
        ("clip_input_faces", lambda c: c.input_faces, np.int32, 3),
        ("clip_oracle_sides", lambda c: c.oracle_sides, np.int16, 1),
        ("clip_cut_points", lambda c: c.cut_points, np.float64, 3),
    ]
    for name, getter, dtype, cols in stream_defs:
        data, offsets = _build_stream(trace.clips, getter, dtype, cols)
        if data.size > 0:
            arrays[name] = data
            arrays[name + "_offsets"] = offsets

    for i, s in enumerate(trace.component_states):
        arrays[f"state_{i}_mesh_verts"] = s.mesh_vertices
        arrays[f"state_{i}_mesh_faces"] = s.mesh_faces
        arrays[f"state_{i}_hull_verts"] = s.hull_vertices
        arrays[f"state_{i}_hull_faces"] = s.hull_faces

    for i, (v, f_arr) in enumerate(trace.output_parts):
        arrays[f"part_{i}_verts"] = v
        arrays[f"part_{i}_faces"] = f_arr

    np.savez(str(out_dir / "arrays.npz"), **arrays)

    with open(out_dir / "costs.json", "w") as f:
        json.dump(
            [
                {
                    "component_id": c.component_id,
                    "rv": c.rv,
                    "hb": c.hb,
                    "energy": c.energy,
                }
                for c in trace.costs
            ],
            f,
        )
    with open(out_dir / "mcts.json", "w") as f:
        json.dump(
            [
                {
                    "iteration": m.iteration,
                    "reward": m.reward,
                    "worst_part": m.worst_part,
                    "visits": m.visits,
                }
                for m in trace.mcts_transitions
            ],
            f,
        )

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_saved_trace(trace_dir: str | Path) -> CoACDTrace:
    """Load a trace saved by save_trace() (CI replay format).

    Supports stream_v3 (concatenated streams) and v2/v1 (per-clip entries).
    """
    trace_dir = Path(trace_dir)

    with open(trace_dir / "meta.json") as f:
        meta = json.load(f)

    npz_path = trace_dir / "arrays.npz"
    npz = np.load(str(npz_path)) if npz_path.exists() else None
    stream = meta.get("format") == "stream_v3" or _is_stream_format(npz)
    streams = _StreamCache(npz) if stream else None

    def _arr(key: str) -> np.ndarray:
        if npz is not None and key in npz:
            return npz[key]
        return np.load(trace_dir / f"{key}.npy")

    trace = CoACDTrace(
        call_id=meta["call_id"],
        input_vertices=_arr("input_verts"),
        input_faces=_arr("input_faces"),
    )

    with open(trace_dir / "planes.json") as f:
        planes_data = json.load(f)
    for p in planes_data:
        trace.planes.append(
            TracedPlane(
                a=p["a"],
                b=p["b"],
                c=p["c"],
                d=p["d"],
                method=p["method"],
                index=p["index"],
            )
        )

    with open(trace_dir / "clips.json") as f:
        clips_data = json.load(f)
    for i, cm in enumerate(clips_data):
        plane = cm["plane"]
        clip = TracedClip(
            component_id=cm["component_id"],
            plane=TracedPlane(
                a=plane["a"],
                b=plane["b"],
                c=plane["c"],
                d=plane["d"],
                method="clip",
                index=-1,
            ),
            pos_verts=cm["pos_verts"],
            pos_faces=cm["pos_faces"],
            neg_verts=cm["neg_verts"],
            neg_faces=cm["neg_faces"],
            intersection_count=cm["intersection_count"],
        )
        if stream and npz is not None:
            clip.pos_vertices = streams.slice("clip_pos_verts", i)
            clip.pos_triangles = streams.slice("clip_pos_faces", i)
            clip.neg_vertices = streams.slice("clip_neg_verts", i)
            clip.neg_triangles = streams.slice("clip_neg_faces", i)
            clip.input_vertices = streams.slice("clip_input_verts", i)
            clip.input_faces = streams.slice("clip_input_faces", i)
            clip.oracle_sides = streams.slice("clip_oracle_sides", i)
            clip.cut_points = streams.slice("clip_cut_points", i)
        elif npz is not None:
            for key, attr in [
                (f"clip_{i}_pos_verts", "pos_vertices"),
                (f"clip_{i}_pos_faces", "pos_triangles"),
                (f"clip_{i}_neg_verts", "neg_vertices"),
                (f"clip_{i}_neg_faces", "neg_triangles"),
                (f"clip_{i}_input_verts", "input_vertices"),
                (f"clip_{i}_input_faces", "input_faces"),
                (f"clip_{i}_oracle_sides", "oracle_sides"),
                (f"clip_{i}_cut_edges", "cut_edges"),
                (f"clip_{i}_cut_points", "cut_points"),
            ]:
                if key in npz:
                    setattr(clip, attr, npz[key])
        else:
            if cm.get("has_meshes"):
                clip.pos_vertices = np.load(trace_dir / f"clip_{i}_pos_verts.npy")
                clip.pos_triangles = np.load(trace_dir / f"clip_{i}_pos_faces.npy")
                clip.neg_vertices = np.load(trace_dir / f"clip_{i}_neg_verts.npy")
                clip.neg_triangles = np.load(trace_dir / f"clip_{i}_neg_faces.npy")
            if cm.get("has_input"):
                clip.input_vertices = np.load(trace_dir / f"clip_{i}_input_verts.npy")
                clip.input_faces = np.load(trace_dir / f"clip_{i}_input_faces.npy")
            if cm.get("has_oracle_sides"):
                clip.oracle_sides = np.load(trace_dir / f"clip_{i}_oracle_sides.npy")
            if cm.get("has_cut_edges"):
                clip.cut_edges = np.load(trace_dir / f"clip_{i}_cut_edges.npy")
            if cm.get("has_cut_points"):
                clip.cut_points = np.load(trace_dir / f"clip_{i}_cut_points.npy")
        trace.clips.append(clip)

    with open(trace_dir / "costs.json") as f:
        costs_data = json.load(f)
    for c in costs_data:
        trace.costs.append(
            TracedCost(
                component_id=c["component_id"],
                rv=c["rv"],
                hb=c["hb"],
                energy=c["energy"],
            )
        )

    with open(trace_dir / "mcts.json") as f:
        mcts_data = json.load(f)
    for m in mcts_data:
        trace.mcts_transitions.append(
            TracedMCTS(
                iteration=m["iteration"],
                reward=m["reward"],
                worst_part=m["worst_part"],
                visits=m["visits"],
            )
        )

    for i in range(meta["num_states"]):
        trace.component_states.append(
            ComponentState(
                iteration=0,
                component_index=i,
                mesh_vertices=_arr(f"state_{i}_mesh_verts"),
                mesh_faces=_arr(f"state_{i}_mesh_faces"),
                hull_vertices=_arr(f"state_{i}_hull_verts"),
                hull_faces=_arr(f"state_{i}_hull_faces"),
            )
        )

    for i in range(meta["num_parts"]):
        verts = _arr(f"part_{i}_verts")
        faces = _arr(f"part_{i}_faces")
        trace.output_parts.append((verts, faces))

    return trace
