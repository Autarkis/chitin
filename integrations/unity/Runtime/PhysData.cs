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

    public class PhysAsset : ScriptableObject
    {
        public int version;
        public int flags;
        public PhysHull[] hulls;
        public PhysBone[] bones;

        public bool HasBones => (flags & 0x01) != 0;
        public bool HasBindPoses => (flags & 0x02) != 0;
    }
}
