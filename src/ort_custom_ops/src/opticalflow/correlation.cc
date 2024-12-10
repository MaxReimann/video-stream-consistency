/*
    Correlation custom layer for ONNXRuntime (CPU)

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

template <typename T>
void correlation_forward(const T*,
    const T*,
    T*,
    bool,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int);

void CorrelationKernel::ComputeCPU(OrtKernelContext* context)
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

    // should be passed in, for future kernels
    // kernel_size = 1, patch_size = 1, stride = 1, padding = 0, dilation = 1, dilation_patch = 1
    const bool correlate_fast = false;  // yes, if the above settings (standard pwcnet settings)
    int kH = 1;
    int kW = 1;
    // int patchH = 1; int patchW = 1;
    // int patchH = 9; int patchW = 9; // sniklas pwcnet/moritz
    int patchH = (2 * max_displacement_ + 1);
    int patchW = patchH;
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

    std::vector<int64_t> output_dims = {N, patchH, patchW, oH, oW};
    auto output = ort_context.GetOutput(0, output_dims.data(), output_dims.size());


    switch (input_X_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            correlation_forward(
                input1.GetTensorData<float>(),
                input2.GetTensorData<float>(),
                output.GetTensorMutableData<float>(),
                correlate_fast,
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
                oW);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            correlation_forward(
                input1.GetTensorData<double>(),
                input2.GetTensorData<double>(),
                output.GetTensorMutableData<double>(),
                correlate_fast,
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
                oW);
            break;
        default:
            throw std::runtime_error("Unsupported input type. Must be float or double.");
    }
}

#define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < H && y >= 0 && y < W)

template <typename scalar_t>
static void correlate_patch(const scalar_t* input1,
    const scalar_t* input2,
    scalar_t* dst,
    int C,
    int iH,
    int iW,
    int kH,
    int kW,
    int dilationH,
    int dilationW,
    int u,
    int v,
    int shiftU,
    int shiftV)
{
    *dst = 0;
    for (int c = 0; c < C; ++c) {
        const int chidx = c * iH * iW;
        for (int i = 0; i < kH; ++i) {
            int i1 = u + i * dilationH;
            int i2 = i1 + shiftU;
            if WITHIN_BOUNDS (i1, i2, iH, iH) {
                for (int j = 0; j < kW; ++j) {
                    int j1 = v + j * dilationW;
                    int j2 = j1 + shiftV;
                    if WITHIN_BOUNDS (j1, j2, iW, iW) {
                        scalar_t v1 = input1[chidx + i1 * iW + j1];  // input1[c][i1][j1];
                        scalar_t v2 = input2[chidx + i2 * iW + j2];  // input2[c][i2][j2];
                        *dst += v1 * v2;
                    }
                }
            }
        }
    }
}

// for setting  kernel_size = 1, patch_size = 1, stride = 1, padding = 0, dilation = 1, dilation_patch = 1
template <typename scalar_t>
static void correlate_patch_fast(const scalar_t* input1,
    const scalar_t* input2,
    scalar_t* dst,
    int C,
    int iH,
    int iW,
    int h,
    int w)
{
    for (int c = 0; c < C; ++c) {
        const int chidx = c * iH * iW;
        scalar_t v1 = input1[chidx + h * iW + w];  // input1[c][i1][j1];
        scalar_t v2 = input2[chidx + h * iW + w];  // input2[c][i2][j2];
        *dst += v1 * v2;
    }
}

template <typename T>
void correlation_forward(const T* input1_acc,
    const T* input2_acc,
    T* output_acc,  // auto output_acc = output.accessor<scalar_t, 5>();
    bool correlate_fast,
    int batch_size,
    int C,
    int iH,
    int iW,
    int kH,
    int kW,
    int patchH,
    int patchW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int dilation_patchH,
    int dilation_patchW,
    int dH,
    int dW,
    int oH,
    int oW)
{
    const int patchRadH = (patchH - 1) / 2;
    const int patchRadW = (patchW - 1) / 2;

    int n, ph, pw, h, w;
#pragma omp parallel for private(n, ph, pw, h, w) collapse(2)
    for (n = 0; n < batch_size; ++n) {
        const int batchidx = n * C * iH * iW;
        const int outbatchidx = n * patchH * patchW * oH * oW;
        for (ph = 0; ph < patchH; ++ph) {
            for (pw = 0; pw < patchW; ++pw) {
                const int phoffset = ph * patchW * oH * oW;
                const int pwoffset = pw * oH * oW;
                for (h = 0; h < oH; ++h) {
                    for (w = 0; w < oW; ++w) {
                        const int hOffset = h * oW;

                        if (correlate_fast) {
                            correlate_patch_fast<T>(&input1_acc[batchidx],
                                &input2_acc[batchidx],
                                &output_acc[outbatchidx + phoffset + pwoffset + hOffset +
                                            w],  // &output_acc[n][ph][pw][h][w],
                                C,
                                iH,
                                iW,
                                h,
                                w);
                        } else {
                            correlate_patch<T>(&input1_acc[batchidx],
                                &input2_acc[batchidx],
                                &output_acc[outbatchidx + phoffset + pwoffset + hOffset +
                                            w],  // &output_acc[n][ph][pw][h][w],
                                C,
                                iH,
                                iW,
                                kH,
                                kW,
                                dilationH,
                                dilationW,
                                -padH + h * dH,
                                -padW + w * dW,
                                (ph - patchRadH) * dilation_patchH,
                                (pw - patchRadW) * dilation_patchW);
                        }
                    }
                }
            }
        }
    }
}
