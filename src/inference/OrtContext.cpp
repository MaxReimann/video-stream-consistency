
#include "OrtContext.h"

#include <onnxruntime_cxx_api.h>


OrtContext::OrtContext()
    : mEnvironment{std::make_unique<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "VideoFlowConsistency")}
    , mEnvironmentRef{*mEnvironment.get()}
{
    mEnvironment->DisableTelemetryEvents();
}

OrtContext::OrtContext(Ort::Env& environment) : mEnvironment{nullptr}, mEnvironmentRef{environment}
{
}

OrtContext::~OrtContext() = default;

OrtContext::OrtContext(OrtContext&&) noexcept = default;

Ort::Env& OrtContext::environment()
{
    return mEnvironmentRef;
}

