from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chitin.result import ExtractionResult


def export_usd(
    result: ExtractionResult, path: str | Path, scene_name: str = "scene"
) -> None:
    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except ImportError:
        raise ImportError(
            "USD output requires usd-core. Install with: pip install chitin[usd]"
        )

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, f"/{scene_name}")
    stage.SetDefaultPrim(root.GetPrim())
    colliders_path = f"/{scene_name}/Colliders"
    UsdGeom.Scope.Define(stage, colliders_path)

    bone_xforms_by_name: dict[str, object] = {}
    if result.bones:
        for b in result.bones:
            safe = b.name.replace("/", "_").replace(" ", "_")
            bone_xforms_by_name[safe] = b.bind_transform

    created_scopes: set[str] = set()
    bone_counters: dict[str, int] = {}

    for i, hull in enumerate(result.hulls):
        if hull.bone_name is not None:
            safe_bone = hull.bone_name.replace("/", "_").replace(" ", "_")
            scope_path = f"{colliders_path}/{safe_bone}"
            if scope_path not in created_scopes:
                xform = UsdGeom.Xform.Define(stage, scope_path)
                if safe_bone in bone_xforms_by_name:
                    mat = bone_xforms_by_name[safe_bone]
                    gf_mat = Gf.Matrix4d(*mat.flatten().tolist())
                    xform.AddTransformOp().Set(gf_mat)
                created_scopes.add(scope_path)
            idx = bone_counters.get(safe_bone, 0)
            bone_counters[safe_bone] = idx + 1
            mesh_path = f"{scope_path}/hull_{idx:04d}"
        else:
            mesh_path = f"{colliders_path}/hull_{i:04d}"

        mesh = UsdGeom.Mesh.Define(stage, mesh_path)

        points = [
            Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in hull.vertices
        ]
        mesh.CreatePointsAttr().Set(points)
        mesh.CreateFaceVertexCountsAttr().Set([3] * (len(hull.indices) // 3))
        mesh.CreateFaceVertexIndicesAttr().Set(hull.indices.tolist())

        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_col.CreateApproximationAttr().Set("convexHull")

    stage.GetRootLayer().Save()
