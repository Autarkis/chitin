using System;
using UnityEngine;

namespace Chitin
{
    [Serializable]
    public struct PhysHull
    {
        public Vector3[] vertices;
        public int[] triangles;
        public Bounds bounds;
        public int boneIndex; // -1 if unassigned
    }

    [Serializable]
    public struct PhysBone
    {
        public string name;
        public Matrix4x4 bindTransform;
    }

    [Serializable]
    public struct PhysLodTier
    {
        public float concavity;
        public PhysHull[] hulls;
    }

    public class PhysAsset : ScriptableObject
    {
        public int version;
        public int flags;
        public PhysHull[] hulls; // LOD 0 (highest detail)
        public PhysBone[] bones;
        public PhysLodTier[] lodTiers; // additional coarser tiers

        public bool HasBones => (flags & 0x01) != 0;
        public bool HasBindPoses => (flags & 0x02) != 0;
        public bool HasLod => (flags & 0x04) != 0;

        // Nearest-concavity tier, or LOD 0 (hulls) when no extra tiers exist.
        public PhysHull[] SelectLod(float concavity)
        {
            if (lodTiers == null || lodTiers.Length == 0) return hulls;
            var best = lodTiers[0];
            foreach (var t in lodTiers)
            {
                if (Math.Abs(t.concavity - concavity) < Math.Abs(best.concavity - concavity))
                    best = t;
            }
            return best.hulls;
        }
    }
}
