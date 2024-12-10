/*
    Correlation custom CUDA layer for ONNXRuntime

    Copyright (C) 2023  Max Reimann (max.reimann@hpi.de)

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
*/

#include <ort_custom_ops/opticalflow/correlation.h>

#include <vector>

#include <cuda_runtime.h>

// clang-format off
template <typename T>
void correlation_cuda_forward(const T*, const T*, T*, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, bool, cudaStream_t compute_stream);
// clang-format on

void CorrelationKernel::ComputeCUDA(OrtKernelContext* context)
{
    Ort::KernelContext ort_context{context};
    Ort::ConstValue input1 = ort_context.GetInput(0);
    Ort::ConstValue input2 = ort_context.GetInput(1);
    Ort::TensorTypeAndShapeInfo input_X_info = input1.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType input_X_type = input_X_info.GetElementType();

    // OrtTensorDimensions dimensions(ort_, input1);
    std::vector<int64_t> dimensions = input_X_info.GetShape();
    const int64_t N = dimensions[0];
    const int64_t C = dimensions[1];
    const int64_t iH = dimensions[2];
    const int64_t iW = dimensions[3];
    
    // std::cout << " dims" << " N: " << N << " C: " << C << " H: " << iH << " W: " << iW << std::endl;

    // should be passed in, for future kernels
    // kernel_size = 1, patch_size = 1, stride = 1, padding = 0, dilation = 1, dilation_patch = 1
    int kH = 1;
    int kW = 1;
    // int patchH = 1; int patchW = 1;
    int patchH = (2 * max_displacement_ + 1);
    int patchW = patchH;
    // int patchW = 9; // sniklas pwcnet/moritz
    int dilationH = 1;
    int dilationW = 1;
    int padH = 0;
    int padW = 0;
    int dH = 1;
    int dW = 1;  // stride
    int dilation_patchH = 1;
    int dilation_patchW = 1;

    const int dilatedKH = (kH - 1) * dilationH + 1;
    const int dilatedKW = (kW - 1) * dilationW + 1;

    const auto oH = (iH + 2 * padH - dilatedKH) / dH + 1;
    const auto oW = (iW + 2 * padW - dilatedKW) / dW + 1;

    std::vector<int64_t> output_dims;

    if (use_legacy_corr_) {
        output_dims = {N, patchH * patchW, oH, oW};
    } else {
        output_dims = {N, patchH, patchW, oH, oW};
    }

    auto output = ort_context.GetOutput(0, output_dims.data(), output_dims.size());

    // important to get current computestream, otherwise, on repeated executions corrupted data will be silently read
    auto compute_stream = reinterpret_cast<cudaStream_t>(ort_context.GetGPUComputeStream());

    switch (input_X_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            correlation_cuda_forward(
                input1.GetTensorData<float>(),
                input2.GetTensorData<float>(),
                output.GetTensorMutableData<float>(),
                N,
                C,
                iH,
                iW,
                kH,
                kW,
                patchH,
                patchW,
                padH,
                padW,
                dilationH,
                dilationW,
                dilation_patchH,
                dilation_patchW,
                dH,
                dW,
                oH,
                oW,
                !use_legacy_corr_,
                compute_stream);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            correlation_cuda_forward(
                input1.GetTensorData<double>(),
                input2.GetTensorData<double>(),
                output.GetTensorMutableData<double>(),
                N,
                C,
                iH,
                iW,
                kH,
                kW,
                patchH,
                patchW,
                padH,
                padW,
                dilationH,
                dilationW,
                dilation_patchH,
                dilation_patchW,
                dH,
                dW,
                oH,
                oW,
                !use_legacy_corr_,
                compute_stream);
            break;
        default:
            throw std::runtime_error("Unsupported input type. Must be float or double.");
    }
}