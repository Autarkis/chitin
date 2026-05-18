using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace Chitin.Editor
{
    [ScriptedImporter(1, "phys")]
    public class PhysImporter : ScriptedImporter
    {
        public bool generateMeshColliders = true;
        public bool attachToSkeleton = true;

        public override void OnImportAsset(AssetImportContext ctx)
        {
            var data = File.ReadAllBytes(ctx.assetPath);
            var phys = PhysReader.Read(data);
            phys.name = Path.GetFileNameWithoutExtension(ctx.assetPath);
            ctx.AddObjectToAsset("phys", phys);

            var root = new GameObject(phys.name + "_colliders");
            ctx.AddObjectToAsset("root", root);
            ctx.SetMainObject(root);

            if (!generateMeshColliders)
                return;

            if (phys.HasBones && phys.bones.Length > 0)
                BuildRiggedColliders(ctx, root, phys);
            else
                BuildStaticColliders(ctx, root, phys);
        }

        void BuildStaticColliders(AssetImportContext ctx, GameObject root, PhysAsset phys)
        {
            for (int i = 0; i < phys.hulls.Length; i++)
            {
                var mesh = HullToMesh(phys.hulls[i], $"hull_{i}");
                ctx.AddObjectToAsset($"mesh_{i}", mesh);

                var child = new GameObject($"hull_{i}");
                child.transform.SetParent(root.transform, false);

                var mc = child.AddComponent<MeshCollider>();
                mc.sharedMesh = mesh;
                mc.convex = true;
            }
        }

        void BuildRiggedColliders(AssetImportContext ctx, GameObject root, PhysAsset phys)
        {
            var boneGroups = new Dictionary<int, GameObject>();

            for (int b = 0; b < phys.bones.Length; b++)
            {
                var boneObj = new GameObject(phys.bones[b].name);
                boneObj.transform.SetParent(root.transform, false);

                var m = phys.bones[b].bindTransform;
                boneObj.transform.localPosition = m.GetColumn(3);
                boneObj.transform.localRotation = m.rotation;
                boneObj.transform.localScale = m.lossyScale;

                boneGroups[b] = boneObj;
            }

            for (int i = 0; i < phys.hulls.Length; i++)
            {
                ref var hull = ref phys.hulls[i];
                var mesh = HullToMesh(hull, $"hull_{i}");
                ctx.AddObjectToAsset($"mesh_{i}", mesh);

                Transform parent = root.transform;
                if (hull.boneIndex >= 0 && boneGroups.TryGetValue(hull.boneIndex, out var boneObj))
                    parent = boneObj.transform;

                var child = new GameObject($"hull_{i}");
                child.transform.SetParent(parent, false);

                var mc = child.AddComponent<MeshCollider>();
                mc.sharedMesh = mesh;
                mc.convex = true;
            }
        }

        static Mesh HullToMesh(PhysHull hull, string name)
        {
            var mesh = new Mesh { name = name };
            mesh.SetVertices(hull.vertices);
            mesh.SetTriangles(hull.triangles, 0);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }
    }
}
