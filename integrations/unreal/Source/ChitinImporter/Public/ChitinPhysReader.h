#pragma once

#include "CoreMinimal.h"

class UChitinPhysAsset;

class CHITINIMPORTER_API FChitinPhysReader
{
public:
    static UChitinPhysAsset* ReadFromFile(const FString& FilePath, UObject* Outer);
    static UChitinPhysAsset* ReadFromBuffer(const TArray<uint8>& Data, UObject* Outer);
};
