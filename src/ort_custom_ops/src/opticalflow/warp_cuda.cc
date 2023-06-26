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
    const OrtValue* tensorinput = ort_.KernelContext_GetInput(context, 0);
    const OrtValue* flowinput = ort_.KernelContext_GetInput(context, 1);

    OrtTensorTypeAndShapeInfo* input_X_info = ort_.GetTensorTypeAndShape(tensorinput);
    ONNXTensorElementDataType input_X_type = ort_.GetTensorElementType(input_X_info);
    ort_.ReleaseTensorTypeAndShapeInfo(input_X_info);

    OrtTensorDimensions dimensions(ort_, tensorinput);
    const int64_t N = dimensions[0];
    const int64_t C = dimensions[1];
    const int64_t iH = dimensions[2];
    const int64_t iW = dimensions[3];

    std::vector<int64_t> output_dims = {N, C, iH, iW};
    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, output_dims.data(), output_dims.size());

    // important to get current computestream, otherwise, on repeated executions corrupted data will be silently read
    auto compute_stream = reinterpret_cast<cudaStream_t>(ort_.KernelContext_GetGPUComputeStream(context));

    switch (input_X_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            warp_cuda_forward(ort_.GetTensorData<float>(tensorinput),
                ort_.GetTensorData<float>(flowinput),
                ort_.GetTensorMutableData<float>(output),
                N,
                C,
                iW,
                iH,
                compute_stream);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            warp_cuda_forward(ort_.GetTensorData<double>(tensorinput),
                ort_.GetTensorData<double>(flowinput),
                ort_.GetTensorMutableData<double>(output),
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
