
#pragma once

#include <ort_custom_ops/basekernel.h>

#include <exception>
#include <iostream>

#include <onnxruntime_c_api.h>

struct WarpKernel : BaseKernel {
public:
    WarpKernel(const OrtApi& api, const OrtKernelInfo* info, const char* provider)
        : BaseKernel(api, info, provider)
    {
    }

    void Compute(OrtKernelContext* context)
    {
        if (std::string(provider_).find("CUDA") != std::string::npos) {
            // std::cout << "warp provider gpu " << std::string(provider_) << std::endl;
#if D_BUILD_WITH_CUDA
            ComputeCUDA(context);
#else
            std::cerr << "ORT requested cuda warp op, but NMP was compiled without CUDA" << std::endl;
#endif
        } else {
            // std::cout << "warp provider gpu " << std::string(provider_) << std::endl;
            ComputeCPU(context);
        }
    }

protected:
    void ComputeCPU(OrtKernelContext* context);
    void ComputeCUDA(OrtKernelContext* context);
};

struct FlowWarpCustomOp : Ort::CustomOpBase<FlowWarpCustomOp, WarpKernel> {
    explicit FlowWarpCustomOp(const char* provider) : provider_(provider) { }

    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new WarpKernel(api, info, provider_);
    };

    const char* GetName() const { return "Warp"; };
    const char* GetExecutionProviderType() const { return provider_; };

    size_t GetInputTypeCount() const { return 2; };
    ONNXTensorElementDataType GetInputType(size_t /*index*/) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;  // input array of float
    };

    size_t GetOutputTypeCount() const { return 1; };
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

private:
    const char* provider_;
};
