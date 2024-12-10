/*
    Warping custom layer for ONNXRuntime (CUDA)

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

#include <ort_custom_ops/opticalflow/warp.h>

#include <cmath>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

template <typename T>
void warp_cuda_forward(const T*, const T*, T*, int, int, int, int, cudaStream_t compute_stream);

void WarpKernel::ComputeCUDA(OrtKernelContext* context)
{
    Ort::KernelContext ort_context{context};
    Ort::ConstValue tensorinput = ort_context.GetInput(0);
    Ort::ConstValue flowinput = ort_context.GetInput(1);
    Ort::TensorTypeAndShapeInfo input_X_info = tensorinput.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType input_X_type = input_X_info.GetElementType();

    // OrtTensorDimensions dimensions(ort_, input1);
    std::vector<int64_t> dimensions = input_X_info.GetShape();
    const int64_t N = dimensions[0];
    const int64_t C = dimensions[1];
    const int64_t iH = dimensions[2];
    const int64_t iW = dimensions[3];

    std::vector<int64_t> output_dims = {N, C, iH, iW};
    // OrtValue* output = ort_.KernelContext_GetOutput(context, 0, output_dims.data(), output_dims.size());
    auto output = ort_context.GetOutput(0, output_dims.data(), output_dims.size());

    // important to get current computestream, otherwise, on repeated executions corrupted data will be silently read
    auto compute_stream = reinterpret_cast<cudaStream_t>(ort_context.GetGPUComputeStream());


    switch (input_X_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            warp_cuda_forward(
                tensorinput.GetTensorData<float>(),
                flowinput.GetTensorData<float>(),
                output.GetTensorMutableData<float>(),
                N,
                C,
                iW,
                iH,
                compute_stream);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            warp_cuda_forward(
                tensorinput.GetTensorData<double>(),
                flowinput.GetTensorData<double>(),
                output.GetTensorMutableData<double>(),
                N,
                C,
                iW,
                iH,
                compute_stream);
            break;
        default:
            throw std::runtime_error("Unsupported input type. Must be float or double.");
    }
}
