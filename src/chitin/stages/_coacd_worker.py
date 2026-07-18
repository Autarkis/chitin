"""Subprocess worker that runs CoACD in isolation so a native stall can be
killed by a timeout in the parent. Reads an input .npz, writes hulls to an
output .npz. Kept import-light (coacd + numpy only) for fast startup."""

from __future__ import annotations

import sys

import numpy as np


def main() -> int:
    in_path, out_path = sys.argv[1], sys.argv[2]
    data = np.load(in_path)
    vertices = data["vertices"].astype(np.float64)
    faces = data["faces"].astype(np.int32)
    threshold = float(data["threshold"][0])
    preprocess_mode = str(data["preprocess_mode"][0])
    preprocess_resolution = int(data["preprocess_resolution"][0])
    max_convex_hull = int(data["max_convex_hull"][0])

    import coacd

    coacd.set_log_level("error")
    mesh = coacd.Mesh(vertices, faces)
    parts = coacd.run_coacd(
        mesh,
        threshold=threshold,
        preprocess_mode=preprocess_mode,
        preprocess_resolution=preprocess_resolution,
        max_convex_hull=max_convex_hull,
    )

    out: dict[str, np.ndarray] = {"n": np.array([len(parts)], dtype=np.int64)}
    for i, (pv, pt) in enumerate(parts):
        out[f"v{i}"] = np.asarray(pv, dtype=np.float32)
        out[f"t{i}"] = np.asarray(pt, dtype=np.uint32)
    np.savez(out_path, **out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
