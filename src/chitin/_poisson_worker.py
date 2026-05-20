# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
"""Subprocess worker for crash-isolated Poisson reconstruction."""

import sys

import numpy as np
import open3d as o3d


def main() -> None:
    in_path, out_path = sys.argv[1], sys.argv[2]
    data = np.load(in_path)

    positions = data["positions"]
    normals = data["normals"] if "normals" in data else None
    depth = int(data["depth"][0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)

    if normals is not None and not np.allclose(normals, 0):
        pcd.normals = o3d.utility.Vector3dVector(normals)
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=10)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    densities = np.asarray(densities)
    if len(densities) > 0:
        density_threshold = np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(densities < density_threshold)

    np.savez(
        out_path,
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        triangles=np.asarray(mesh.triangles, dtype=np.int32),
    )


if __name__ == "__main__":
    main()
