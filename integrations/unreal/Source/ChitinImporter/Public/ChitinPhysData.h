#pragma once

#include "CoreMinimal.h"
#include "ChitinPhysData.generated.h"

USTRUCT(BlueprintType)
struct FChitinHull
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    TArray<FVector> Vertices;

    UPROPERTY(BlueprintReadOnly)
    TArray<int32> Indices;

    UPROPERTY(BlueprintReadOnly)
    FBox Bounds;

    UPROPERTY(BlueprintReadOnly)
    int32 BoneIndex = -1;
};

USTRUCT(BlueprintType)
struct FChitinBone
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    FString Name;

    UPROPERTY(BlueprintReadOnly)
    FMatrix BindTransform;
};

USTRUCT(BlueprintType)
struct FChitinLodTier
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    float Concavity = 0.0f;

    UPROPERTY(BlueprintReadOnly)
    TArray<FChitinHull> Hulls;
};

UCLASS(BlueprintType)
class CHITINIMPORTER_API UChitinPhysAsset : public UObject
{
    GENERATED_BODY()

public:
    UPROPERTY(BlueprintReadOnly)
    int32 Version = 0;

    UPROPERTY(BlueprintReadOnly)
    int32 Flags = 0;

    // LOD 0 (highest detail).
    UPROPERTY(BlueprintReadOnly)
    TArray<FChitinHull> Hulls;

    UPROPERTY(BlueprintReadOnly)
    TArray<FChitinBone> Bones;

    // Additional coarser tiers, ascending concavity.
    UPROPERTY(BlueprintReadOnly)
    TArray<FChitinLodTier> LodTiers;

    bool HasBones() const { return (Flags & 0x01) != 0; }
    bool HasBindPoses() const { return (Flags & 0x02) != 0; }
    bool HasLod() const { return (Flags & 0x04) != 0; }

    // Nearest-concavity tier, or LOD 0 (Hulls) when no extra tiers exist.
    const TArray<FChitinHull>& SelectLod(float Concavity) const
    {
        if (LodTiers.Num() == 0) return Hulls;
        int32 Best = 0;
        for (int32 i = 1; i < LodTiers.Num(); i++)
        {
            if (FMath::Abs(LodTiers[i].Concavity - Concavity) <
                FMath::Abs(LodTiers[Best].Concavity - Concavity))
                Best = i;
        }
        return LodTiers[Best].Hulls;
    }
};
