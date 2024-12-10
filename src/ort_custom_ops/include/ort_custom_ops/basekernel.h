
#pragma once

#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT
#else
#include <onnxruntime_cxx_api.h>
#endif


struct BaseKernel {
    BaseKernel(const OrtApi& api) : info_(nullptr), ort_(api) { }
    BaseKernel(const OrtApi& api, const OrtKernelInfo* info) : info_(info), ort_(api) { }
    BaseKernel(const OrtApi& api, const OrtKernelInfo* info, const char* provider)
        : info_(info)
        , ort_(api)
        , provider_(provider)
    {
    }

protected:
    OrtApi ort_;
    const OrtKernelInfo* info_;
    const char* provider_;
};