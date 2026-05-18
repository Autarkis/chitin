#include "ChitinPhysReader.h"
#include "ChitinPhysData.h"
#include "Misc/FileHelper.h"

static constexpr uint32 PhysMagic = 0x53594850; // "PHYS" LE
static constexpr int32 HeaderSize = 32;
static constexpr int32 FlagHasBones = 0x01;
static constexpr int32 FlagHasBindPoses = 0x02;

static float Dequantize(int16 Q, float Min, float Extent)
{
    return ((Q + 32768.0f) / 65535.0f) * Extent + Min;
}

UChitinPhysAsset* FChitinPhysReader::ReadFromFile(const FString& FilePath, UObject* Outer)
{
    TArray<uint8> Data;
    if (!FFileHelper::LoadFileToArray(Data, *FilePath))
    {
        UE_LOG(LogTemp, Error, TEXT("Chitin: failed to read %s"), *FilePath);
        return nullptr;
    }
    return ReadFromBuffer(Data, Outer);
}

UChitinPhysAsset* FChitinPhysReader::ReadFromBuffer(const TArray<uint8>& Data, UObject* Outer)
{
    if (Data.Num() < HeaderSize)
    {
        UE_LOG(LogTemp, Error, TEXT("Chitin: file too small"));
        return nullptr;
    }

    const uint8* Ptr = Data.GetData();
    FMemoryReader Ar(Data, true);

    uint32 Magic;
    Ar << Magic;
    if (Magic != PhysMagic)
    {
        UE_LOG(LogTemp, Error, TEXT("Chitin: bad magic 0x%08X"), Magic);
        return nullptr;
    }

    uint16 Version, Flags;
    Ar << Version;
    Ar << Flags;

    uint32 HullCount, TotalVerts, TotalIdx;
    Ar << HullCount;
    Ar << TotalVerts;
    Ar << TotalIdx;

    uint32 HullTableOff, VertexDataOff, IndexDataOff;
    Ar << HullTableOff;
    Ar << VertexDataOff;
    Ar << IndexDataOff;

    bool bHasBones = (Flags & FlagHasBones) != 0;
    int32 DescSize = bHasBones ? 44 : 40;

    struct FHullDesc
    {
        uint32 VOffset, VCount, IOffset, ICount;
        FVector3f AabbMin, AabbMax;
        int32 BoneIndex;
    };

    TArray<FHullDesc> Descs;
    Descs.SetNum(HullCount);

    Ar.Seek(HullTableOff);
    for (uint32 i = 0; i < HullCount; i++)
    {
        FHullDesc& D = Descs[i];
        Ar << D.VOffset;
        Ar << D.VCount;
        Ar << D.IOffset;
        Ar << D.ICount;
        Ar << D.AabbMin.X << D.AabbMin.Y << D.AabbMin.Z;
        Ar << D.AabbMax.X << D.AabbMax.Y << D.AabbMax.Z;
        D.BoneIndex = -1;
        if (bHasBones)
            Ar << D.BoneIndex;
    }

    UChitinPhysAsset* Asset = NewObject<UChitinPhysAsset>(Outer);
    Asset->Version = Version;
    Asset->Flags = Flags;
    Asset->Hulls.SetNum(HullCount);

    for (uint32 i = 0; i < HullCount; i++)
    {
        const FHullDesc& D = Descs[i];
        FChitinHull& Hull = Asset->Hulls[i];
        Hull.BoneIndex = D.BoneIndex;

        FVector3f Extent = D.AabbMax - D.AabbMin;
        for (int32 C = 0; C < 3; C++)
        {
            if (Extent[C] == 0.0f) Extent[C] = 1.0f;
        }

        Hull.Vertices.SetNum(D.VCount);
        Ar.Seek(VertexDataOff + D.VOffset * 6);
        for (uint32 V = 0; V < D.VCount; V++)
        {
            int16 Qx, Qy, Qz;
            Ar << Qx << Qy << Qz;
            Hull.Vertices[V] = FVector(
                Dequantize(Qx, D.AabbMin.X, Extent.X),
                Dequantize(Qy, D.AabbMin.Y, Extent.Y),
                Dequantize(Qz, D.AabbMin.Z, Extent.Z)
            );
        }

        Hull.Indices.SetNum(D.ICount);
        Ar.Seek(IndexDataOff + D.IOffset * 2);
        for (uint32 T = 0; T < D.ICount; T++)
        {
            uint16 Idx;
            Ar << Idx;
            Hull.Indices[T] = Idx;
        }

        FVector Min(D.AabbMin.X, D.AabbMin.Y, D.AabbMin.Z);
        FVector Max(D.AabbMax.X, D.AabbMax.Y, D.AabbMax.Z);
        Hull.Bounds = FBox(Min, Max);
    }

    if ((Flags & FlagHasBindPoses) != 0)
    {
        Ar.Seek(IndexDataOff + TotalIdx * 2);
        uint32 BoneCount;
        Ar << BoneCount;
        Asset->Bones.SetNum(BoneCount);

        for (uint32 B = 0; B < BoneCount; B++)
        {
            FChitinBone& Bone = Asset->Bones[B];

            FMatrix M;
            for (int32 Row = 0; Row < 4; Row++)
                for (int32 Col = 0; Col < 4; Col++)
                {
                    float Val;
                    Ar << Val;
                    M.M[Row][Col] = Val;
                }
            Bone.BindTransform = M;

            uint16 NameLen;
            Ar << NameLen;
            TArray<uint8> NameBytes;
            NameBytes.SetNum(NameLen);
            Ar.Serialize(NameBytes.GetData(), NameLen);
            Bone.Name = FString(UTF8_TO_TCHAR(
                StringCast<TCHAR>(
                    reinterpret_cast<const UTF8CHAR*>(NameBytes.GetData()),
                    NameLen
                ).Get()
            ));
        }
    }

    return Asset;
}
