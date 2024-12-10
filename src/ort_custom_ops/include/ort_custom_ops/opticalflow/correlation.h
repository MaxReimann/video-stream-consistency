
#pragma once

#include <ort_custom_ops/basekernel.h>

#include <iostream>

#include <onnxruntime_c_api.h>

struct CorrelationKernel : BaseKernel {
public:
    CorrelationKernel(const OrtApi& api, const OrtKernelInfo* info, const char* provider)
        : BaseKernel(api, info, provider)
    {
        // obtain attribute from onnx node
        // use_legacy_corr_ = ort_.KernelInfoGetAttribute<int64_t>(info, "legacy");
        // max_displacement_ = ort_.KernelInfoGetAttribute<int64_t>(info, "max_displacement");
        
        int64_t legacy;
        OrtStatusPtr err = ort_.KernelInfoGetAttribute_int64(info, "legacy", &legacy);
        Ort::Status status(err);
        if (!status.IsOK()) {
            throw std::runtime_error("Error reading attribute 'legacy', with error: " + status.GetErrorMessage());
        }
        this->use_legacy_corr_ = legacy;

        err = ort_.KernelInfoGetAttribute_int64(info, "max_displacement", &(this->max_displacement_));
        Ort::Status status2(err);
        if (!status2.IsOK()) {
            throw std::runtime_error("Error reading attribute 'max_displacement', with error: " + status2.GetErrorMessage());
        }

    }

    void Compute(OrtKernelContext* context)
    {
        if (std::string(provider_).find("CUDA") != std::string::npos) {
            // std::cout << "corr provider gpu " << std::string(provider_) << std::endl;
#if D_BUILD_WITH_CUDA
            ComputeCUDA(context);
#else
            std::cerr << "ORT requested cuda correlation op, but NMP was compiled without CUDA" << std::endl;
#endif
        } else {
            if (use_legacy_corr_)
                throw std::runtime_error("Legacy correlation not supported in CPU mode (only cuda)");
            // std::cout << "corr provider cpu " << std::string(provider_) << std::endl;
            ComputeCPU(context);
        }
    }

protected:
    void ComputeCPU(OrtKernelContext* context);
    void ComputeCUDA(OrtKernelContext* context);

    bool use_legacy_corr_;
    int64_t max_displacement_;
};

struct FlowCorrelationCustomOp : Ort::CustomOpBase<FlowCorrelationCustomOp, CorrelationKernel> {
    explicit FlowCorrelationCustomOp(const char* provider) : provider_(provider) { }

    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new CorrelationKernel(api, info, provider_);
    };

    const char* GetName() const { return "Correlation"; };
    const char* GetExecutionProviderType() const { return provider_; };

    size_t GetInputTypeCount() const { return 2; };
    ONNXTensorElementDataType GetInputType(size_t /*index*/) const
    {
        // // Both the inputs need to be necessarily of float type
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;  // input array of float or double, the two input frames, as float
                                                     // tensor
    };

    size_t GetOutputTypeCount() const { return 1; };
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

private:
    const char* provider_;
};
