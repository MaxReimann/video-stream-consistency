
#pragma once

#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT
#else
#include <onnxruntime_cxx_api.h>
#endif

struct OrtTensorDimensions : std::vector<int64_t> {
    OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value)
    {
        OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
        std::vector<int64_t>::operator=(ort.GetTensorShape(info));
        ort.ReleaseTensorTypeAndShapeInfo(info);
    }
    const std::vector<int64_t>& GetDims() const { return *this; }
    int64_t Size() const
    {
        int64_t s = 1.;
        for (auto it = begin(); it != end(); ++it)
            s *= *it;
        return s;
    }
};

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
    Ort::CustomOpApi ort_;
    const OrtKernelInfo* info_;
    const char* provider_;
};