"""Cap triangulation from welded boundary loops.

CPU-side, GPU-adjacent: consumes the same WeldResult/LoopExtractionResult
data GPU dispatch produces, and fills the same role a GPU triangulation
kernel eventually would (see chitin.gpu.loops for the analogous CPU
boundary-walk stand-in).

Uses boundary-preserving dynamic programming in the permitted host cap
stage. The gate validates coverage independently of triangulation choices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chitin.cap_triangulation import triangulate_polygon
from chitin.gpu.weld import WeldResult


@dataclass
class CapResult:
    positive_cap_faces: np.ndarray  # (Fp, 3) uint32
    negative_cap_faces: np.ndarray  # (Fn, 3) uint32
    loop_count: int
    total_cap_faces: int
    valid: bool


def triangulate_caps(
    loops: list[np.ndarray],
    weld_result: WeldResult,
    source_vertices: np.ndarray,
    plane_normal: np.ndarray,
) -> CapResult:
    """Triangulate closed boundary loops into cap faces.

    `loops` is a list of ordered vertex-index rings (source vertices
    below len(source_vertices), welded intersection vertices at or
    above it -- the same convention chitin.gpu.loops produces). Loops
    shorter than a triangle make the whole result invalid: a degenerate
    loop means upstream boundary extraction didn't hand back a usable
    ring, and a partial cap is worse than none.
    """
    empty = np.zeros((0, 3), dtype=np.uint32)

    if not loops or any(len(l_data) < 3 for l_data in loops):
        return CapResult(
            positive_cap_faces=empty,
            negative_cap_faces=empty,
            loop_count=len(loops),
            total_cap_faces=0,
            valid=False,
        )

    full_vertices = np.concatenate(
        [source_vertices.astype(np.float32), weld_result.welded_positions]
    )

    drop = int(np.argmax(np.abs(plane_normal)))
    keep = [j for j in range(3) if j != drop]

    all_faces: list[list[int]] = []
    pn64 = plane_normal.astype(np.float64)
    for loop in loops:
        p2 = full_vertices[loop][:, keep]
        try:
            loop_faces = triangulate_polygon(loop, p2)
        except ValueError:
            return CapResult(empty.copy(), empty.copy(), len(loops), 0, False)
        if loop_faces:
            tri = full_vertices[np.array(loop_faces[0], dtype=np.uint32)]
            cap_normal = np.cross(
                tri[1].astype(np.float64) - tri[0].astype(np.float64),
                tri[2].astype(np.float64) - tri[0].astype(np.float64),
            )
            if np.dot(cap_normal, pn64) < 0:
                loop_faces = [[f[0], f[2], f[1]] for f in loop_faces]
        all_faces.extend(loop_faces)

    if all_faces:
        positive_cap_faces = np.array(all_faces, dtype=np.uint32)
    else:
        positive_cap_faces = empty.copy()

    negative_cap_faces = positive_cap_faces.copy()
    if len(negative_cap_faces) > 0:
        negative_cap_faces[:, [1, 2]] = negative_cap_faces[:, [2, 1]]

    return CapResult(
        positive_cap_faces=positive_cap_faces,
        negative_cap_faces=negative_cap_faces,
        loop_count=len(loops),
        total_cap_faces=len(positive_cap_faces),
        valid=True,
    )
