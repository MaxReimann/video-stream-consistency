/*
    Spatial correlation for cost volume computation. Contains two corr implementations, with differing output ordering
    1) "correlation_old_kernel" based on corr of original PWCNet, https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/
    2 ) "correlation_cuda_forward_kernel" newer version by spatial correlation sampler
    https://github.com/ClementPinard/Pytorch-Correlation-extension

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
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < H && y >= 0 && y < W)

#define THREADS_FORWARD 32

// == rearrangem channels last Kernel
template <typename scalar_t>
__global__ void blob_rearrange_kernel(const scalar_t* in,
    scalar_t* out,
    int num,
    int channels,
    int width,
    int height,
    int widthheight,
    int padding,
    int pwidthheight)
{
    int xy = blockIdx.x * blockDim.x + threadIdx.x;
    if (xy >= widthheight)
        return;

    int ch = blockIdx.y;
    int n = blockIdx.z;

    scalar_t value = in[(n * channels + ch) * widthheight + xy];
    // out[(n * channels + ch) * widthheight + xy] = 0.0;

    __syncthreads();

    int xpad = (xy % width + padding);
    int ypad = (xy / width + padding);
    int xypad = ypad * (width + 2 * padding) + xpad;

    out[(n * pwidthheight + xypad) * channels + ch] = value;
    // out[n * pwidthheight * channels + ch] = value;
}

template <typename scalar_t>
void blob_rearrange_ongpu(const scalar_t* in,
    scalar_t* out,
    int num,
    int channels,
    int width,
    int height,
    int widthheight,
    int padding,
    int pwidthheight,
    cudaStream_t stream)
{
    int threads_per_block = 16;
    dim3 totalBlocksRearr((widthheight - 1) / threads_per_block + 1, channels, num);

    cudaError_t err;

    blob_rearrange_kernel<<<totalBlocksRearr, threads_per_block, 0, stream>>>(in,
        out,
        num,
        channels,
        width,
        height,
        widthheight,
        padding,
        pwidthheight);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

template <typename scalar_t>
__global__ void correlation_cuda_forward_kernel(const scalar_t* rInput1,
    const scalar_t* rInput2,
    scalar_t* output,
    int N,
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
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    const int w = blockIdx.z;
    const int thread = threadIdx.x;

    const int start_i = -padH + h * dH;
    const int start_j = -padW + w * dW;

    const int patchRadH = dilation_patchH * (patchH - 1) / 2;
    const int patchRadW = dilation_patchW * (patchW - 1) / 2;

    __shared__ scalar_t prod_sum[THREADS_FORWARD];

    const int batchidx = n * C * iH * iW;
    const int outbatchidx = n * patchH * patchW * oH * oW;

    for (int ph = 0; ph < patchH; ++ph) {
        int ph_dilated = ph * dilation_patchH - patchRadH;
        for (int pw = 0; pw < patchW; ++pw) {
            int pw_dilated = pw * dilation_patchW - patchRadW;
            prod_sum[thread] = 0;
            for (int i = 0; i < kH; ++i) {
                int i1 = start_i + i * dilationH;
                int i2 = i1 + ph_dilated;
                int i1Offset = i1 * iW * C;
                int i2Offset = i2 * iW * C;
                if WITHIN_BOUNDS (i1, i2, iH, iH) {
                    for (int j = 0; j < kW; ++j) {
                        int j1 = start_j + j * dilationW;
                        int j2 = j1 + pw_dilated;
                        if WITHIN_BOUNDS (j1, j2, iW, iW) {
                            for (int c = thread; c < C; c += THREADS_FORWARD) {
                                // scalar_t v1 = rInput1[n][i1][j1][c];
                                scalar_t v1 = rInput1[batchidx + i1Offset + j1 * C + c];
                                // scalar_t v2 = rInput2[n][i2][j2][c];
                                scalar_t v2 = rInput2[batchidx + i2Offset + j2 * C + c];
                                prod_sum[thread] += v1 * v2;
                            }
                        }
                    }
                }
            }
            // accumulate
            __syncthreads();
            if (thread == 0) {
                scalar_t reduce_sum = 0;
                for (int index = 0; index < THREADS_FORWARD; ++index) {
                    reduce_sum += prod_sum[index];
                }
                // output[n][ph][pw][h][w] = reduce_sum;
                int phOffset = oH * oW * patchW * ph;
                output[outbatchidx + phOffset + oH * oW * pw + oW * h + w] = reduce_sum;
            }
        }
    }
}

#define CUDA_NUM_THREADS 1024
#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32
// correlation from original PWCNet Code
// https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/external_packages/correlation-pytorch-master/correlation-pytorch/correlation_package/src/corr_cuda_kernel.cu
template <typename scalar_t>
__global__ void correlation_old_kernel(const int nthreads,
    int num,
    int topwidth,
    int topheight,
    int topchannels,
    int topcount,
    int max_displacement,
    int neighborhood_grid_radius,
    int neighborhood_grid_width,
    int kernel_radius,
    int kernel_size,
    int stride1,
    int stride2,
    int bottomwidth,
    int bottomheight,
    int bottomchannels,
    const scalar_t* bottom0,
    const scalar_t* bottom1,
    scalar_t* top)
{
    extern __shared__ char patch_data_char[];

    scalar_t* patch_data = (scalar_t*)patch_data_char;

    // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
    int x1 = blockIdx.x * stride1 + max_displacement;
    int y1 = blockIdx.y * stride1 + max_displacement;
    int item = blockIdx.z;
    int ch_off = threadIdx.x;

    // Load 3D patch into shared shared memory
    for (int j = 0; j < kernel_size; j++) {  // HEIGHT
        for (int i = 0; i < kernel_size; i++) {  // WIDTH
            int ji_off = ((j * kernel_size) + i) * bottomchannels;
            for (int ch = ch_off; ch < bottomchannels; ch += (WARPS_PER_BLOCK * THREADS_PER_WARP)) {  // CHANNELS
                int idx1 = ((item * bottomheight + y1 + j) * bottomwidth + x1 + i) * bottomchannels + ch;
                int idxPatchData = ji_off + ch;
                patch_data[idxPatchData] = bottom0[idx1];
            }
        }
    }

    __syncthreads();

    __shared__ scalar_t sum[WARPS_PER_BLOCK * THREADS_PER_WARP];

    // Compute correlation
    for (int top_channel = 0; top_channel < topchannels; top_channel++) {
        sum[ch_off] = 0;

        int s2o = (top_channel % neighborhood_grid_width - neighborhood_grid_radius) * stride2;
        int s2p = (top_channel / neighborhood_grid_width - neighborhood_grid_radius) * stride2;

        for (int j = 0; j < kernel_size; j++) {  // HEIGHT
            for (int i = 0; i < kernel_size; i++) {  // WIDTH
                int ji_off = ((j * kernel_size) + i) * bottomchannels;
                for (int ch = ch_off; ch < bottomchannels; ch += (WARPS_PER_BLOCK * THREADS_PER_WARP)) {  // CHANNELS
                    int x2 = x1 + s2o;
                    int y2 = y1 + s2p;

                    int idxPatchData = ji_off + ch;
                    int idx2 = ((item * bottomheight + y2 + j) * bottomwidth + x2 + i) * bottomchannels + ch;

                    sum[ch_off] += patch_data[idxPatchData] * bottom1[idx2];
                }
            }
        }

        __syncthreads();

        if (ch_off == 0) {
            scalar_t total_sum = 0;
            for (int idx = 0; idx < WARPS_PER_BLOCK * THREADS_PER_WARP; idx++) {
                total_sum += sum[idx];
            }
            const int sumelems = kernel_size * kernel_size * bottomchannels;
            const int index = ((top_channel * topheight + blockIdx.y) * topwidth) + blockIdx.x;
            top[index + item * topcount] = total_sum / (scalar_t)sumelems;
        }
    }

    // Aggregate
}

template <typename scalar_t>
void CorrelateData_old(const scalar_t* rbot1,
    const scalar_t* rbot2,
    scalar_t* output,
    int N,
    int inC,
    int oH,
    int oW,
    int patch_size,
    int kernel_size,
    int dH,
    int dW,
    int paddedbottomwidth,
    int paddedbottomheight,
    cudaStream_t stream)
{
    int max_displacement = (patch_size - 1) / 2;

    // Given a center position in image 1, how many displaced positions in -x / +x
    // direction do we consider in image2 (neighborhood_grid_width)
    int neighborhood_grid_radius_ = max_displacement / dW;
    int neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;

    // Number of output channels amounts to displacement combinations in X and Y direction
    int nOutputPlane = neighborhood_grid_width_ * neighborhood_grid_width_;

    dim3 threadsPerBlock(THREADS_PER_WARP * WARPS_PER_BLOCK);
    // 4 * inC take from sniklaus
    int shared_memory_per_block = (kernel_size * kernel_size) * inC;
    int outputCount = oH * oW * nOutputPlane;
    int outputThreadCount = outputCount;
    dim3 totalBlocksCorr(oW, oH, N);

    long kernel_radius_ = (kernel_size - 1) / 2;
    // long border_size_ = max_displacement + kernel_radius_; // size of unreachable border region (on each side)

    cudaError_t err;

    correlation_old_kernel<<<totalBlocksCorr, threadsPerBlock, shared_memory_per_block * sizeof(scalar_t), stream>>>(
        outputThreadCount,
        N,
        oW,
        oH,
        nOutputPlane,
        outputCount,
        max_displacement,
        neighborhood_grid_radius_,
        neighborhood_grid_width_,
        kernel_radius_,
        kernel_size,
        dH,
        dW,
        paddedbottomwidth,
        paddedbottomheight,
        inC,
        rbot1,
        rbot2,
        output);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

template <typename scalar_t>
void correlation_cuda_forward(const scalar_t* rInput1,
    const scalar_t* rInput2,
    scalar_t* output,
    int N,
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
    int oW,
    // use for networks trained with spatial-correlation-sampler package, if False uses PWCNet original corr code
    bool spatial_corr_sampler,
    cudaStream_t compute_stream)
{
    int __padH = 4;
    int __padW = 4;
    int paddedbottomheight = iH + 2 * __padH;
    int paddedbottomwidth = iW + 2 * __padW;
    int pwidthheight = paddedbottomwidth * paddedbottomheight;

    scalar_t *rbot1_data, *rbot2_data;  // device tmp buffer
    // Allocate space for device tmp buffers
    int wh = spatial_corr_sampler ? (iH * iW) : pwidthheight;
    int size = wh * N * C * sizeof(scalar_t);
    cudaMalloc((void**)&rbot1_data, size);
    cudaMalloc((void**)&rbot2_data, size);

    blob_rearrange_ongpu(rInput1,
        rbot1_data,
        N,
        C,
        iW,
        iH,
        iW * iH,
        spatial_corr_sampler ? padH : __padH,
        wh,
        compute_stream);
    blob_rearrange_ongpu(rInput2,
        rbot2_data,
        N,
        C,
        iW,
        iH,
        iW * iH,
        spatial_corr_sampler ? padH : __padH,
        wh,
        compute_stream);

    if (spatial_corr_sampler) {
        const dim3 blocks(N, oH, oW);
        cudaError_t err;
        correlation_cuda_forward_kernel<<<blocks, THREADS_FORWARD, 0, compute_stream>>>(rbot1_data,
            rbot2_data,
            output,
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

        err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    } else {
        CorrelateData_old(rbot1_data,
            rbot2_data,
            output,
            N,
            C,
            oH,
            oW,
            patchW,
            kW,
            dH,
            dW,
            paddedbottomwidth,
            paddedbottomheight,
            compute_stream);
    }
    // Cleanup
    cudaFree(rbot1_data);
    cudaFree(rbot2_data);
}

template void correlation_cuda_forward(const float*,
    const float*,
    float*,
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
    int,
    bool,
    cudaStream_t compute_stream);
template void correlation_cuda_forward(const double*,
    const double*,
    double*,
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
    int,
    bool,
    cudaStream_t compute_stream);