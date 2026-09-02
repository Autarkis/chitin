"""Capture CoACD trace corpus for f32 gate CI replay.

Run with the traced DLL deployed:
    python scripts/capture_trace_corpus.py

Saves to tests/fixtures/traces/<name>/
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chitin.coacd_trace import capture_trace, save_trace

CORPUS_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "traces"


def make_box(sx=1.0, sy=1.0, sz=1.0):
    """Axis-aligned box."""
    v = np.array(
        [
            [0, 0, 0],
            [sx, 0, 0],
            [sx, sy, 0],
            [0, sy, 0],
            [0, 0, sz],
            [sx, 0, sz],
            [sx, sy, sz],
            [0, sy, sz],
        ],
        dtype=np.float64,
    )
    f = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int32,
    )
    return v, f


def make_l_shape():
    """L-shaped extrusion (non-convex)."""
    v = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [2, 1, 0],
            [1, 1, 0],
            [1, 2, 0],
            [0, 2, 0],
            [0, 0, 1],
            [2, 0, 1],
            [2, 1, 1],
            [1, 1, 1],
            [1, 2, 1],
            [0, 2, 1],
        ],
        dtype=np.float64,
    )
    f = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [6, 8, 7],
            [6, 9, 8],
            [6, 10, 9],
            [6, 11, 10],
            [0, 7, 1],
            [0, 6, 7],
            [1, 8, 2],
            [1, 7, 8],
            [2, 9, 3],
            [2, 8, 9],
            [3, 10, 4],
            [3, 9, 10],
            [4, 11, 5],
            [4, 10, 11],
            [5, 6, 0],
            [5, 11, 6],
        ],
        dtype=np.int32,
    )
    return v, f


def make_t_shape():
    """T-shaped extrusion (non-convex, different topology from L)."""
    v = np.array(
        [
            # Bottom bar
            [0, 0, 0],
            [3, 0, 0],
            [3, 1, 0],
            [0, 1, 0],
            # Top stem
            [1, 1, 0],
            [2, 1, 0],
            [2, 3, 0],
            [1, 3, 0],
            # Extruded z=1
            [0, 0, 1],
            [3, 0, 1],
            [3, 1, 1],
            [0, 1, 1],
            [1, 1, 1],
            [2, 1, 1],
            [2, 3, 1],
            [1, 3, 1],
        ],
        dtype=np.float64,
    )
    f = np.array(
        [
            # Bottom face
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            # Top face
            [8, 10, 9],
            [8, 11, 10],
            [12, 14, 13],
            [12, 15, 14],
            # Sides - bottom bar
            [0, 9, 1],
            [0, 8, 9],
            [1, 10, 2],
            [1, 9, 10],
            [2, 11, 3],
            [2, 10, 11],
            [3, 8, 0],
            [3, 11, 8],
            # Sides - stem
            [4, 13, 5],
            [4, 12, 13],
            [5, 14, 6],
            [5, 13, 14],
            [6, 15, 7],
            [6, 14, 15],
            [7, 12, 4],
            [7, 15, 12],
        ],
        dtype=np.int32,
    )
    return v, f


def make_thin_panel():
    """Thin panel — stress-tests classification near plane."""
    v = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 0.01],
            [1, 0, 0.01],
            [1, 1, 0.01],
            [0, 1, 0.01],
        ],
        dtype=np.float64,
    )
    f = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int32,
    )
    return v, f


def make_staircase():
    """3-step staircase — multiple concavities."""
    steps = []
    for i in range(3):
        y0 = i * 0.5
        z0 = i * 0.5
        bv = np.array(
            [
                [0, y0, z0],
                [1, y0, z0],
                [1, y0 + 0.5, z0],
                [0, y0 + 0.5, z0],
                [0, y0, z0 + 0.5],
                [1, y0, z0 + 0.5],
                [1, y0 + 0.5, z0 + 0.5],
                [0, y0 + 0.5, z0 + 0.5],
            ],
            dtype=np.float64,
        )
        steps.append(bv)

    # Merge into single mesh
    all_v = np.vstack(steps)
    all_f = []
    box_faces = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
    ]
    for i in range(3):
        off = i * 8
        for face in box_faces:
            all_f.append([face[0] + off, face[1] + off, face[2] + off])
    return all_v, np.array(all_f, dtype=np.int32)


def make_high_complexity_sphere(subdivisions=3):
    """Icosphere with many faces — stress-tests plane count."""
    import math

    phi = (1 + math.sqrt(5)) / 2
    verts = [
        [-1, phi, 0],
        [1, phi, 0],
        [-1, -phi, 0],
        [1, -phi, 0],
        [0, -1, phi],
        [0, 1, phi],
        [0, -1, -phi],
        [0, 1, -phi],
        [phi, 0, -1],
        [phi, 0, 1],
        [-phi, 0, -1],
        [-phi, 0, 1],
    ]
    # Normalize to unit sphere
    verts = [np.array(v) / np.linalg.norm(v) for v in verts]
    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]
    for _ in range(subdivisions):
        new_faces = []
        midpoint_cache = {}

        def midpoint(i1, i2):
            key = (min(i1, i2), max(i1, i2))
            if key in midpoint_cache:
                return midpoint_cache[key]
            p = (np.array(verts[i1]) + np.array(verts[i2])) / 2
            p = p / np.linalg.norm(p)
            idx = len(verts)
            verts.append(p)
            midpoint_cache[key] = idx
            return idx

        for f in faces:
            a, b, c = f
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        faces = new_faces

    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


FIXTURES = {
    "box": make_box,
    "l_shape": make_l_shape,
    "t_shape": make_t_shape,
    "thin_panel": make_thin_panel,
    "staircase": make_staircase,
    "icosphere": make_high_complexity_sphere,
}

# Import new fixtures from trace_fixtures (single source of truth)
from chitin.trace_fixtures import FIXTURES as _TF  # noqa: E402

for _name in (
    "thin_u_channel",
    "curved_pipe_quarter",
    "cross_bracket",
    "h_shape",
    "nested_box",
):
    if _name in _TF:
        FIXTURES[_name] = _TF[_name]


def capture_single(name: str):
    """Capture one fixture (called as subprocess to get fresh DLL state)."""
    gen = FIXTURES[name]
    verts, faces = gen()
    out = CORPUS_DIR / name
    trace = capture_trace(verts, faces, threshold=0.05)
    save_trace(trace, out)
    print(
        f"{name}: clips={len(trace.clips)}, planes={len(trace.planes)}, "
        f"mcts={len(trace.mcts_transitions)}, parts={len(trace.output_parts)}"
    )


def main():
    import subprocess

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    for name in FIXTURES:
        print(f"Capturing {name}...", flush=True)
        result = subprocess.run(
            [sys.executable, __file__, name],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"  FAILED: {result.stderr.strip()}")
        else:
            print(f"  {result.stdout.strip()}")

    print(f"\nCorpus saved to {CORPUS_DIR}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        capture_single(sys.argv[1])
    else:
        main()
