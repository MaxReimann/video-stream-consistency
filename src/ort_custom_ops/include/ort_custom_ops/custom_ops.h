
#pragma once

#include <onnxruntime_c_api.h>


#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT
#else
#include <onnxruntime_cxx_api.h>
#endif

#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

namespace nmp {

class CustomOpRegistry {
public:
    static CustomOpRegistry& getInstance()
    {
        static CustomOpRegistry instance;  // Instantiated on first use.
        return instance;
    }

private:
    Ort::CustomOpDomain m_customdomain;
    bool isRegistered;

    CustomOpRegistry() : m_customdomain(Ort::CustomOpDomain("custom")), isRegistered(false) { }

public:
    // singleton pattern: delete unwanted methods
    CustomOpRegistry(CustomOpRegistry const&) = delete;
    void operator=(CustomOpRegistry const&) = delete;
};

}

// If used from python, load the nmp lib as dynamic lib
#ifdef __cplusplus
extern "C" {
#endif

EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);

#ifdef __cplusplus
}
#endif
