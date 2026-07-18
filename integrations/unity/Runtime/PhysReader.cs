using System;
using System.IO;
using System.Text;
using UnityEngine;

namespace Chitin
{
    public static class PhysReader
    {
        const uint Magic = 0x53594850; // "PHYS" LE
        const int HeaderSize = 32;
        const int FlagHasBones = 0x01;
        const int FlagHasBindPoses = 0x02;
        const int FlagHasLod = 0x04;
        const int LodTierHeaderSize = 24;

        public static PhysAsset Read(byte[] data)
        {
            using var stream = new MemoryStream(data);
            using var r = new BinaryReader(stream);
            return Read(r);
        }

        public static PhysAsset Read(string path)
        {
            var data = File.ReadAllBytes(path);
            return Read(data);
        }

        static PhysAsset Read(BinaryReader r)
        {
            uint magic = r.ReadUInt32();
            if (magic != Magic)
                throw new InvalidDataException($"Bad magic: 0x{magic:X8}");

            ushort version = r.ReadUInt16();
            ushort flags = r.ReadUInt16();
            uint hullCount = r.ReadUInt32();
            uint totalVerts = r.ReadUInt32();
            uint totalIdx = r.ReadUInt32();
            uint hullTableOff = r.ReadUInt32();
            uint vertexDataOff = r.ReadUInt32();
            uint indexDataOff = r.ReadUInt32();

            bool hasBones = (flags & FlagHasBones) != 0;
            int descSize = hasBones ? 44 : 40;

            var hulls = ReadHulls(r, hullTableOff, hullCount, vertexDataOff, indexDataOff, hasBones);

            long nextBlock = indexDataOff + (long)totalIdx * 2;

            PhysBone[] bones = Array.Empty<PhysBone>();
            if ((flags & FlagHasBindPoses) != 0)
            {
                r.BaseStream.Position = nextBlock;
                uint boneCount = r.ReadUInt32();
                bones = new PhysBone[boneCount];

                for (int b = 0; b < boneCount; b++)
                {
                    var m = new Matrix4x4();
                    for (int row = 0; row < 4; row++)
                        for (int col = 0; col < 4; col++)
                            m[row, col] = r.ReadSingle();
                    m = Matrix4x4.Transpose(m);

                    ushort nameLen = r.ReadUInt16();
                    string name = Encoding.UTF8.GetString(r.ReadBytes(nameLen));

                    bones[b] = new PhysBone { name = name, bindTransform = m };
                }
                nextBlock = r.BaseStream.Position;
            }

            PhysLodTier[] lodTiers = Array.Empty<PhysLodTier>();
            if ((flags & FlagHasLod) != 0)
            {
                r.BaseStream.Position = nextBlock;
                uint tierCount = r.ReadUInt32();
                lodTiers = new PhysLodTier[tierCount];

                for (int tier = 0; tier < tierCount; tier++)
                {
                    float concavity = r.ReadSingle();
                    uint tHullCount = r.ReadUInt32();
                    uint tTotalVerts = r.ReadUInt32();
                    r.ReadUInt32(); // total indices (not needed for layout)
                    uint dataSize = r.ReadUInt32();
                    r.ReadUInt32(); // reserved

                    long tHullTableOff = r.BaseStream.Position;
                    long tVertexDataOff = tHullTableOff + (long)tHullCount * descSize;
                    long tIndexDataOff = tVertexDataOff + (long)tTotalVerts * 6;

                    lodTiers[tier] = new PhysLodTier
                    {
                        concavity = concavity,
                        hulls = ReadHulls(r, tHullTableOff, tHullCount, tVertexDataOff, tIndexDataOff, hasBones),
                    };

                    r.BaseStream.Position = tHullTableOff + dataSize;
                }
            }

            var asset = ScriptableObject.CreateInstance<PhysAsset>();
            asset.version = version;
            asset.flags = flags;
            asset.hulls = hulls;
            asset.bones = bones;
            asset.lodTiers = lodTiers;
            return asset;
        }

        // Reads one hull table (LOD 0 or a LOD tier). Descriptors are contiguous
        // from hullTableOff; vertex/index data live at the block's data offsets.
        static PhysHull[] ReadHulls(BinaryReader r, long hullTableOff, uint hullCount,
            long vertexDataOff, long indexDataOff, bool hasBones)
        {
            var descriptors = new HullDescriptor[hullCount];
            r.BaseStream.Position = hullTableOff;

            for (int i = 0; i < hullCount; i++)
            {
                descriptors[i] = new HullDescriptor
                {
                    vertexOffset = r.ReadUInt32(),
                    vertexCount = r.ReadUInt32(),
                    indexOffset = r.ReadUInt32(),
                    indexCount = r.ReadUInt32(),
                    aabbMin = new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle()),
                    aabbMax = new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle()),
                    boneIndex = hasBones ? r.ReadInt32() : -1,
                };
            }

            var hulls = new PhysHull[hullCount];
            for (int i = 0; i < hullCount; i++)
            {
                ref var desc = ref descriptors[i];

                r.BaseStream.Position = vertexDataOff + desc.vertexOffset * 6;
                var verts = new Vector3[desc.vertexCount];
                Vector3 extent = desc.aabbMax - desc.aabbMin;
                for (int c = 0; c < 3; c++)
                {
                    if (extent[c] == 0) extent[c] = 1f;
                }

                for (int v = 0; v < desc.vertexCount; v++)
                {
                    float x = Dequantize(r.ReadInt16(), desc.aabbMin.x, extent.x);
                    float y = Dequantize(r.ReadInt16(), desc.aabbMin.y, extent.y);
                    float z = Dequantize(r.ReadInt16(), desc.aabbMin.z, extent.z);
                    verts[v] = new Vector3(x, y, z);
                }

                r.BaseStream.Position = indexDataOff + desc.indexOffset * 2;
                var tris = new int[desc.indexCount];
                for (int t = 0; t < desc.indexCount; t++)
                    tris[t] = r.ReadUInt16();

                hulls[i] = new PhysHull
                {
                    vertices = verts,
                    triangles = tris,
                    bounds = new Bounds(
                        (desc.aabbMin + desc.aabbMax) * 0.5f,
                        desc.aabbMax - desc.aabbMin
                    ),
                    boneIndex = desc.boneIndex,
                };
            }

            return hulls;
        }

        static float Dequantize(short q, float min, float extent)
        {
            return ((q + 32768f) / 65535f) * extent + min;
        }

        struct HullDescriptor
        {
            public uint vertexOffset, vertexCount, indexOffset, indexCount;
            public Vector3 aabbMin, aabbMax;
            public int boneIndex;
        }
    }
}
