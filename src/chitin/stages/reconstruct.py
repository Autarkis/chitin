from __future__ import annotations

from pathlib import Path

import numpy as np

from chitin.resolve import ResolvedConfig

_POISSON_WORKER_SCRIPT = Path(__file__).parent.parent / "_poisson_worker.py"


class PoissonWorkerError(RuntimeError):
    """Isolated Poisson worker failed; message carries exit code and stderr tail."""


def poisson_reconstruct_inner(
    positions: np.ndarray,
    normals: np.ndarray | None,
    depth: int,
    density_quantile: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "Point cloud extraction requires open3d. "
            "Install with: pip install chitin[splat]"
        )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)

    if normals is not None and not np.allclose(normals, 0):
        pcd.normals = o3d.utility.Vector3dVector(normals)
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=10)

    # n_threads=1: cell-level parallelism already comes from the process
    # pool; per-cell OpenMP pools oversubscribe cores, segfault sporadically
    # under contention (exit -11), and make float accumulation order -- and
    # thus the output mesh -- nondeterministic.
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=1
    )

    densities = np.asarray(densities)
    if len(densities) > 0:
        density_threshold = np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(densities < density_threshold)

    return (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.triangles, dtype=np.int32),
    )


# Open3D's Poisson reconstruction can segfault nondeterministically at high
# octree depths. Depths at or above this run in a subprocess so a crash is
# contained to one reconstruction rather than killing the compiler process.
RISKY_POISSON_DEPTH = 8


def poisson_reconstruct(
    positions: np.ndarray,
    normals: np.ndarray | None,
    config: ResolvedConfig,
    isolate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    depth = config.poisson_depth
    dq = config.poisson_density_quantile

    # Force subprocess isolation for risky manual depths even when the caller
    # requested in-process reconstruction.
    if depth is not None and depth >= RISKY_POISSON_DEPTH:
        isolate = True

    if not isolate:
        return poisson_reconstruct_inner(positions, normals, depth, dq)

    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "input.npz"
        out_path = Path(tmpdir) / "output.npz"

        save_dict: dict[str, np.ndarray] = {
            "positions": positions,
            "depth": np.array([depth]),
            "density_quantile": np.array([dq]),
        }
        if normals is not None:
            save_dict["normals"] = normals
        np.savez(in_path, **save_dict)

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(_POISSON_WORKER_SCRIPT),
                    str(in_path),
                    str(out_path),
                ],
                capture_output=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired as exc:
            raise PoissonWorkerError("worker timeout after 300s") from exc

        if result.returncode != 0 or not out_path.exists():
            stderr_tail = ""
            if result.stderr:
                lines = result.stderr.decode(errors="replace").strip().splitlines()
                if lines:
                    stderr_tail = lines[-1][:160]
            raise PoissonWorkerError(
                f"worker exit {result.returncode}: {stderr_tail or 'no stderr'}"
            )

        # Close the NpzFile handle before the TemporaryDirectory is cleaned up,
        # otherwise the still-open file blocks removal on Windows.
        with np.load(out_path) as data:
            return data["vertices"], data["triangles"]
