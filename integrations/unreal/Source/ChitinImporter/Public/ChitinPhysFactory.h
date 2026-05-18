#pragma once

#include "CoreMinimal.h"
#include "Factories/Factory.h"
#include "ChitinPhysFactory.generated.h"

UCLASS()
class CHITINIMPORTER_API UChitinPhysFactory : public UFactory
{
    GENERATED_BODY()

public:
    UChitinPhysFactory();

    virtual UObject* FactoryCreateBinary(
        UClass* InClass,
        UObject* InParent,
        FName InName,
        EObjectFlags InFlags,
        UObject* Context,
        const TCHAR* Type,
        const uint8*& Buffer,
        const uint8* BufferEnd,
        FFeedbackContext* Warn
    ) override;
};
