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

    begin = events[0]
    call_id = begin["call_id"]
    input_verts = _load_npy(trace_dir, begin["input_verts"])
    input_faces = _load_npy(trace_dir, begin["input_faces"])

    trace = CoACDTrace(
        call_id=call_id,
        input_vertices=input_verts,
        input_faces=input_faces,
    )

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
                filename = ev.get(key, "")
                if filename and isinstance(filename, str):
                    npy_path = trace_dir / filename
                    if npy_path.exists():
                        setattr(clip, attr, np.load(npy_path))
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
            state = ComponentState(
                iteration=ev["iteration"],
                component_index=ev["component_index"],
                mesh_vertices=_load_npy(trace_dir, ev["mesh_verts"]),
                mesh_faces=_load_npy(trace_dir, ev["mesh_faces"]),
                hull_vertices=_load_npy(trace_dir, ev["hull_verts"]),
                hull_faces=_load_npy(trace_dir, ev["hull_faces"]),
            )
            trace.component_states.append(state)
        elif kind == "end":
            part_files = ev["part_files"]
            for i in range(0, len(part_files), 2):
                verts = _load_npy(trace_dir, part_files[i])
                faces = _load_npy(trace_dir, part_files[i + 1])
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


def save_trace(trace: CoACDTrace, out_dir: str | Path) -> None:
    """Serialize a trace to disk for CI replay (no traced DLL needed)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save main arrays
    np.save(out_dir / "input_verts.npy", trace.input_vertices)
    np.save(out_dir / "input_faces.npy", trace.input_faces)

    # Save metadata as JSON
    meta: dict = {
        "call_id": trace.call_id,
        "num_planes": len(trace.planes),
        "num_clips": len(trace.clips),
        "num_costs": len(trace.costs),
        "num_mcts": len(trace.mcts_transitions),
        "num_states": len(trace.component_states),
        "num_parts": len(trace.output_parts),
    }

    # Save planes
    planes_data = [
        {"a": p.a, "b": p.b, "c": p.c, "d": p.d, "method": p.method, "index": p.index}
        for p in trace.planes
    ]
    with open(out_dir / "planes.json", "w") as f:
        json.dump(planes_data, f)

    # Save clips metadata (meshes saved as .npy)
    clips_meta = []
    for i, c in enumerate(trace.clips):
        cm = {
            "component_id": c.component_id,
            "plane": {"a": c.plane.a, "b": c.plane.b, "c": c.plane.c, "d": c.plane.d},
            "pos_verts": c.pos_verts,
            "pos_faces": c.pos_faces,
            "neg_verts": c.neg_verts,
            "neg_faces": c.neg_faces,
            "intersection_count": c.intersection_count,
        }
        if (
            c.pos_vertices is not None
            and c.pos_triangles is not None
            and c.neg_vertices is not None
            and c.neg_triangles is not None
        ):
            np.save(out_dir / f"clip_{i}_pos_verts.npy", c.pos_vertices)
            np.save(out_dir / f"clip_{i}_pos_faces.npy", c.pos_triangles)
            np.save(out_dir / f"clip_{i}_neg_verts.npy", c.neg_vertices)
            np.save(out_dir / f"clip_{i}_neg_faces.npy", c.neg_triangles)
            cm["has_meshes"] = True
        if c.input_vertices is not None and c.input_faces is not None:
            np.save(out_dir / f"clip_{i}_input_verts.npy", c.input_vertices)
            np.save(out_dir / f"clip_{i}_input_faces.npy", c.input_faces)
            cm["has_input"] = True
        if c.oracle_sides is not None:
            np.save(out_dir / f"clip_{i}_oracle_sides.npy", c.oracle_sides)
            cm["has_oracle_sides"] = True
        if c.cut_edges is not None:
            np.save(out_dir / f"clip_{i}_cut_edges.npy", c.cut_edges)
            cm["has_cut_edges"] = True
        if c.cut_points is not None:
            np.save(out_dir / f"clip_{i}_cut_points.npy", c.cut_points)
            cm["has_cut_points"] = True
        clips_meta.append(cm)
    with open(out_dir / "clips.json", "w") as f:
        json.dump(clips_meta, f)

    # Save component states
    for i, s in enumerate(trace.component_states):
        np.save(out_dir / f"state_{i}_mesh_verts.npy", s.mesh_vertices)
        np.save(out_dir / f"state_{i}_mesh_faces.npy", s.mesh_faces)
        np.save(out_dir / f"state_{i}_hull_verts.npy", s.hull_vertices)
        np.save(out_dir / f"state_{i}_hull_faces.npy", s.hull_faces)

    # Save output parts
    for i, (v, f_arr) in enumerate(trace.output_parts):
        np.save(out_dir / f"part_{i}_verts.npy", v)
        np.save(out_dir / f"part_{i}_faces.npy", f_arr)

    # Save costs, mcts as JSON
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
    """Load a trace saved by save_trace() (CI replay format)."""
    trace_dir = Path(trace_dir)

    with open(trace_dir / "meta.json") as f:
        meta = json.load(f)

    trace = CoACDTrace(
        call_id=meta["call_id"],
        input_vertices=np.load(trace_dir / "input_verts.npy"),
        input_faces=np.load(trace_dir / "input_faces.npy"),
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
                mesh_vertices=np.load(trace_dir / f"state_{i}_mesh_verts.npy"),
                mesh_faces=np.load(trace_dir / f"state_{i}_mesh_faces.npy"),
                hull_vertices=np.load(trace_dir / f"state_{i}_hull_verts.npy"),
                hull_faces=np.load(trace_dir / f"state_{i}_hull_faces.npy"),
            )
        )

    for i in range(meta["num_parts"]):
        verts = np.load(trace_dir / f"part_{i}_verts.npy")
        faces = np.load(trace_dir / f"part_{i}_faces.npy")
        trace.output_parts.append((verts, faces))

    return trace
