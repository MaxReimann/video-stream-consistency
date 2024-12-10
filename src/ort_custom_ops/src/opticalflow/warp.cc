/*  
    Warping custom layer for ONNXRuntime (CPU)

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
#include <vector>

template <typename T>
void warp(const T*, const T*, T*, int, int, int, int);

void WarpKernel::ComputeCPU(OrtKernelContext* context)
{
    Ort::KernelContext ort_context{context};
    Ort::ConstValue tensorinput = ort_context.GetInput(0);
    Ort::ConstValue flowinput = ort_context.GetInput(1);
    Ort::TensorTypeAndShapeInfo input_X_info = tensorinput.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType input_X_type = input_X_info.GetElementType();

    std::vector<int64_t> dimensions = input_X_info.GetShape();
    const int64_t N = dimensions[0];
    const int64_t C = dimensions[1];
    const int64_t iH = dimensions[2];
    const int64_t iW = dimensions[3];

    std::vector<int64_t> output_dims = {N, C, iH, iW};
    Ort::UnownedValue output = ort_context.GetOutput(0, output_dims.data(), output_dims.size());

    switch (input_X_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            warp(
                tensorinput.GetTensorData<float>(),
                flowinput.GetTensorData<float>(),
                output.GetTensorMutableData<float>(),
                N,
                C,
                iW,
                iH);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            warp(
                tensorinput.GetTensorData<double>(),
                flowinput.GetTensorData<double>(),
                output.GetTensorMutableData<double>(),
                N,
                C,
                iW,
                iH);
            break;
        default:
            throw std::runtime_error("Unsupported input type. Must be float or double.");
    }
}

#define INPUT(y, x) input[n * input_channels * width * height + c * width * height + y * width + x]
#define FLOW(ch, x, y) flow[n * 2 * width * height + ch * width * height + y * width + x]

template <typename scalar_t>
void warp(const scalar_t* input,
    const scalar_t* flow,
    scalar_t* dst,
    int batch_size,
    int input_channels,
    int width,
    int height)
{
#pragma omp parallel for private(n, c, x, y) collapse(2)
    for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < input_channels; c++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    const scalar_t xf = static_cast<scalar_t>(x) + FLOW(0, x, y);
                    const scalar_t yf = static_cast<scalar_t>(y) + FLOW(1, x, y);
                    // fraction to next pixel
                    const scalar_t alpha = xf - std::floor(xf);
                    const scalar_t beta = yf - std::floor(yf);

                    const scalar_t right_edge = static_cast<scalar_t>(width - 1);
                    const scalar_t bottom_edge = static_cast<scalar_t>(height - 1);

                    const scalar_t xL = std::floor(xf);
                    const scalar_t xR = std::floor(xf) + 1.0;
                    const scalar_t yT = std::floor(yf);
                    const scalar_t yB = std::floor(yf) + 1.0;

                    // boundary checks
                    const int maskL = (0 <= xL && xL <= right_edge) ? 1 : 0;
                    const int maskR = (0 <= xR && xR <= right_edge) ? 1 : 0;
                    const int maskT = (0 <= yT && yT <= bottom_edge) ? 1 : 0;
                    const int maskB = (0 <= yB && yB <= bottom_edge) ? 1 : 0;

                    scalar_t val = 0.0;
                    scalar_t mask = 0.0;

                    // bilinearly sampled mask, to determine fraction that is over tensor edge
                    mask += (1.0 - alpha) * (1.0 - beta) * (maskT + maskL == 2 ? 1.0 : 0.0);
                    mask += (alpha) * (1.0 - beta) * (maskT + maskR == 2 ? 1.0 : 0.0);
                    mask += (1.0 - alpha) * (beta) * (maskB + maskL == 2 ? 1.0 : 0.0);
                    mask += (alpha) * (beta) * (maskB + maskR == 2 ? 1.0 : 0.0);

                    // exclude over-edge values, except for very close values
                    if (mask > 0.999) {
                        // bilinear sampling
                        // emulate constant zero padding of the sampled tensor, by returning 0 if (xf,yf) is not in
                        // bounds
                        val += (1.0 - alpha) * (1.0 - beta) *
                               (maskT + maskL == 2 ? INPUT(static_cast<int>(yT), static_cast<int>(xL)) : 0.0);
                        val += (alpha) * (1.0 - beta) *
                               (maskT + maskR == 2 ? INPUT(static_cast<int>(yT), static_cast<int>(xR)) : 0.0);
                        val += (1.0 - alpha) * (beta) *
                               (maskB + maskL == 2 ? INPUT(static_cast<int>(yB), static_cast<int>(xL)) : 0.0);
                        val += (alpha) * (beta) *
                               (maskB + maskR == 2 ? INPUT(static_cast<int>(yB), static_cast<int>(xR)) : 0.0);
                    }

                    dst[n * input_channels * width * height + c * width * height + y * width + x] = val;
                }
            }
        }
    }
}