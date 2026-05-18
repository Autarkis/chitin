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

UCLASS(BlueprintType)
class CHITINIMPORTER_API UChitinPhysAsset : public UObject
{
    GENERATED_BODY()

public:
    UPROPERTY(BlueprintReadOnly)
    int32 Version = 0;

    UPROPERTY(BlueprintReadOnly)
    int32 Flags = 0;

    UPROPERTY(BlueprintReadOnly)
    TArray<FChitinHull> Hulls;

    UPROPERTY(BlueprintReadOnly)
    TArray<FChitinBone> Bones;

    bool HasBones() const { return (Flags & 0x01) != 0; }
    bool HasBindPoses() const { return (Flags & 0x02) != 0; }
};
