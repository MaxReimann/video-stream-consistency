/*
    Warping custom layer CUDA kernels

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

#include <cstdint>
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_FORWARD 32
#define INPUT(ch, y, x) input[n * oC * oW * oH + ch * oW * oH + y * oW + x]
#define FLOW(ch, x, y) flow[n * 2 * oW * oH + ch * oW * oH + y * oW + x]

template <typename scalar_t>
__global__ void warp_forward_kernel(const scalar_t* input, const scalar_t* flow, scalar_t* dst, int oW, int oH, int oC)
{
    int xy = blockIdx.x * blockDim.x + threadIdx.x;
    if (xy >= oW * oH)
        return;

    int x = xy % oW;
    int y = xy / oW;

    int ch = blockIdx.y;
    int n = blockIdx.z;

    scalar_t xf = static_cast<scalar_t>(x) + FLOW(0, x, y);
    scalar_t yf = static_cast<scalar_t>(y) + FLOW(1, x, y);
    // fraction to next pixel
    scalar_t alpha = xf - floor(xf);
    scalar_t beta = yf - floor(yf);

    scalar_t right_edge = static_cast<scalar_t>(oW - 1);
    scalar_t bottom_edge = static_cast<scalar_t>(oH - 1);

    scalar_t xL = floor(xf);
    scalar_t xR = floor(xf) + 1.0;
    scalar_t yT = floor(yf);
    scalar_t yB = floor(yf) + 1.0;

    // boundary checks
    int maskL = (0 <= xL && xL <= right_edge) ? 1 : 0;
    int maskR = (0 <= xR && xR <= right_edge) ? 1 : 0;
    int maskT = (0 <= yT && yT <= bottom_edge) ? 1 : 0;
    int maskB = (0 <= yB && yB <= bottom_edge) ? 1 : 0;

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
        // emulate constant zero padding of the sampled tensor, by returning 0 if (xf,yf) is not in bounds
        val += (1.0 - alpha) * (1.0 - beta) *
               (maskT + maskL == 2 ? INPUT(ch, static_cast<int>(yT), static_cast<int>(xL)) : 0.0);
        val +=
            (alpha) * (1.0 - beta) * (maskT + maskR == 2 ? INPUT(ch, static_cast<int>(yT), static_cast<int>(xR)) : 0.0);
        val +=
            (1.0 - alpha) * (beta) * (maskB + maskL == 2 ? INPUT(ch, static_cast<int>(yB), static_cast<int>(xL)) : 0.0);
        val += (alpha) * (beta) * (maskB + maskR == 2 ? INPUT(ch, static_cast<int>(yB), static_cast<int>(xR)) : 0.0);
    }

    dst[n * oC * oW * oH + ch * oW * oH + y * oW + x] = val;
}

template <typename scalar_t>
void warp_cuda_forward(const scalar_t* input1,
    const scalar_t* flow,
    scalar_t* output,
    int N,
    int C,
    int oW,
    int oH,
    cudaStream_t compute_stream)
{
    dim3 totalBlocksRearr((oH * oW - 1) / THREADS_FORWARD + 1, C, N);
    warp_forward_kernel<<<totalBlocksRearr, THREADS_FORWARD, 0, compute_stream>>>(input1, flow, output, oW, oH, C);
}

template void warp_cuda_forward(const float*, const float*, float*, int, int, int, int, cudaStream_t compute_stream);
template void warp_cuda_forward(const double*, const double*, double*, int, int, int, int, cudaStream_t compute_stream);
