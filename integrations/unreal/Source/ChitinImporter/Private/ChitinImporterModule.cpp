#include "Modules/ModuleManager.h"

class FChitinImporterModule : public IModuleInterface
{
public:
    virtual void StartupModule() override {}
    virtual void ShutdownModule() override {}
};

IMPLEMENT_MODULE(FChitinImporterModule, ChitinImporter)
