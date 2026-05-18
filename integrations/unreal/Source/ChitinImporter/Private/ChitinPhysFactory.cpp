#include "ChitinPhysFactory.h"
#include "ChitinPhysData.h"
#include "ChitinPhysReader.h"

UChitinPhysFactory::UChitinPhysFactory()
{
    bCreateNew = false;
    bEditorImport = true;
    SupportedClass = UChitinPhysAsset::StaticClass();
    Formats.Add(TEXT("phys;Chitin Physics Colliders"));
}

UObject* UChitinPhysFactory::FactoryCreateBinary(
    UClass* InClass,
    UObject* InParent,
    FName InName,
    EObjectFlags InFlags,
    UObject* Context,
    const TCHAR* Type,
    const uint8*& Buffer,
    const uint8* BufferEnd,
    FFeedbackContext* Warn)
{
    TArray<uint8> Data;
    Data.Append(Buffer, BufferEnd - Buffer);

    UChitinPhysAsset* Asset = FChitinPhysReader::ReadFromBuffer(Data, InParent);
    if (!Asset)
    {
        UE_LOG(LogTemp, Error, TEXT("Chitin: failed to parse .phys file"));
        return nullptr;
    }

    Asset->Rename(*InName.ToString(), InParent);
    Asset->SetFlags(InFlags);
    return Asset;
}
