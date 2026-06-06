"""Subprocess worker for crash-isolated Poisson reconstruction."""

import sys

import numpy as np


def main() -> None:
    in_path, out_path = sys.argv[1], sys.argv[2]
    data = np.load(in_path)

    positions = data["positions"]
    normals = data["normals"] if "normals" in data else None
    depth = int(data["depth"][0])
    density_quantile = (
        float(data["density_quantile"][0]) if "density_quantile" in data else 0.1
    )

    from chitin.stages.reconstruct import poisson_reconstruct_inner

    vertices, triangles = poisson_reconstruct_inner(
        positions, normals, depth, density_quantile
    )

    np.savez(out_path, vertices=vertices, triangles=triangles)


if __name__ == "__main__":
    main()
