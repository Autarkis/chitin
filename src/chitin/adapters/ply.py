# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

from pathlib import Path

import numpy as np

from chitin.adapters import AdapterResult


def load_ply(path: Path) -> AdapterResult:
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    positions = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(
        np.float64
    )

    opacity = None
    has_opacity = "opacity" in vertex.data.dtype.names
    if has_opacity:
        opacity = np.asarray(vertex["opacity"], dtype=np.float64)

    has_scales = all(f"scale_{i}" in vertex.data.dtype.names for i in range(3))
    has_rots = all(f"rot_{i}" in vertex.data.dtype.names for i in range(4))
    has_covariance = has_scales and has_rots

    normals = None
    scales = None
    rots = None
    if has_covariance:
        scales = np.column_stack([vertex[f"scale_{i}"] for i in range(3)]).astype(
            np.float64
        )
        rots = np.column_stack([vertex[f"rot_{i}"] for i in range(4)]).astype(
            np.float64
        )
    else:
        has_normals = all(n in vertex.data.dtype.names for n in ("nx", "ny", "nz"))
        if has_normals:
            normals = np.column_stack(
                [vertex["nx"], vertex["ny"], vertex["nz"]]
            ).astype(np.float64)

    return AdapterResult(
        positions=positions,
        format="ply",
        normals=normals,
        scales=scales,
        rots=rots,
        opacity=opacity,
        detected={
            "has_opacity": has_opacity,
            "has_covariance": has_covariance,
            "has_normals": normals is not None or has_covariance,
        },
    )
