/*  The CUDA kernels for flow-based temporal consistency

    Copyright (C) 2023 Moritz Hilscher and Sumit Shekhar (sumit.shekhar@hpi.de)

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
*/


#include "flowconsistency.cuh"
#include "gpuimagewrapper.h"

#define CUDA(image) GPUImageWrapper((image))

#include <iostream>
#include <string>

void checkError(std::string stage) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error at " << stage << ": " << error << ", " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

//////////////////////////////////////// CUDA Kernels ///////////////////////////////////////////////

__global__ void kernel_average_frames(GPUImageWrapper frame1, GPUImageWrapper frame2, GPUImageWrapper out) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    if (ix >= out.width || iy >= out.height || iy < 0 || ix < 0) {
        return;
    }

    for (int c = 0; c < 3; c++) {
        float v1 = frame1.read(ix, iy, c);
        float v2 = frame2.read(ix, iy, c);
        out.write(ix, iy, c, 0.5*v1 + 0.5*v2);
    }
}

__global__ void kernel_bilinear(GPUImageWrapper input, GPUImageWrapper output) {
    const int ox = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int oy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    if (ox >= output.width || oy >= output.height || ox < 0 || oy < 0) {
        return;
    }

    // input image coordinates
    float xx = (ox * float(input.width) / float(output.width));
    float yy = (oy * float(input.height) / float(output.height));
    int ix = floor(xx);
    int iy = floor(yy);
    float fx = xx - ix;
    float fy = yy - iy;

    // https://en.wikipedia.org/wiki/Bilinear_interpolation#Unit_square
    for (int c = 0; c < output.channels; c++) {
        float v00 = input.read(ix, iy, c);
        float v10 = input.read(min(ix+1, input.width-1), iy, c);
        float v01 = input.read(ix, min(iy+1, input.height-1), c);
        float v11 = input.read(min(ix+1, input.width-1), min(iy+1, input.height-1), c);
        float v = v00*(1-fx)*(1-fy) + v10*fx*(1-fy) + v01*(1-fx)*fy + v11*fx*fy;
        output.write(ox, oy, c, v);
    }
}

__global__ void kernel_warp(GPUImageWrapper input, GPUImageWrapper flow, GPUImageWrapper input_warp) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    if (ix >= input_warp.width || iy >= input_warp.height || iy < 0 || ix < 0) {
        return;
    }

    float out_val, flo_x, flo_y, flo_fx, flo_fy, map_fx, map_fy;
    int flo_ix, flo_iy, map_ix, map_iy;
    float tmp_1, tmp_2;

    // reading the flow value
    flo_x = flow.read(ix, iy, 0);
    flo_y = flow.read(ix, iy, 1);

    // computing the corresponding coordinates
    // floating-point correspondence
    map_fx = max(0.0f, min( (ix + flo_x), float(input_warp.width - 3) ));
    map_fy = max(0.0f, min( (iy + flo_y), float(input_warp.height - 3) ));
    // integer correspondence coordinate
    map_ix = floor(map_fx);
    map_iy = floor(map_fy);
    // fractional correspondence coordinate
    flo_fx = map_fx - map_ix;
    flo_fy = map_fy - map_iy;

    for (int c = 0; c < 3; c++) {

        // performing bilinear interpolation
        tmp_1 = input.read(map_ix , map_iy, c)*(1.0f - flo_fx) + input.read(map_ix + 1, map_iy, c)*flo_fx;
        tmp_2 = input.read(map_ix , map_iy + 1, c)*(1.0f - flo_fx) + input.read(map_ix + 1, map_iy + 1, c)*flo_fx;

        out_val = tmp_1*(1.0f - flo_fy) + tmp_2*flo_fy;

        input_warp.write(ix, iy, c, out_val);
    }
}

__global__ void kernel_adap_comb(GPUImageWrapper crntIn, GPUImageWrapper crntPr, GPUImageWrapper prevWarpIn, GPUImageWrapper prevWarpPr,
                                 GPUImageWrapper nextWarpIn, GPUImageWrapper nextWarpPr, GPUImageWrapper adapCmbIn, GPUImageWrapper adapCmbPr,
                                 GPUImageWrapper lastStabWarp, float alpha){

    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    if (ix >= crntIn.width || iy >= crntIn.height || iy < 0 || ix < 0) {
        return;
    }

    float crnt_val_in, crnt_val_pr, prev_val_in, prev_val_pr, nxt_val_in, nxt_val_pr;
    float wt_prv, wt_nxt, wt_stb, adp_val_in, adp_val_pr, last_stab_val;

    for (int c = 0; c < 3; c++) {

        //reading input values
        crnt_val_in = crntIn.read(ix, iy, c);

        prev_val_in = prevWarpIn.read(ix, iy, c);
        nxt_val_in  = nextWarpIn.read(ix, iy, c);

        //reading per-frame processed values
        crnt_val_pr = crntPr.read(ix, iy, c);

        prev_val_pr = prevWarpPr.read(ix, iy, c);
        nxt_val_pr  = nextWarpPr.read(ix, iy, c);

        last_stab_val = lastStabWarp.read(ix, iy, c);

        wt_prv = expf(-alpha*(crnt_val_in - prev_val_in)*(crnt_val_in - prev_val_in));
        wt_nxt = expf(-alpha*(crnt_val_in - nxt_val_in)*(crnt_val_in - nxt_val_in));

        //global-backward and local-forward consistency
        if(wt_prv > 0.45f) wt_prv = 0.45f; // a higher value of wt_prv increase stability. however for fast motions it also results in ghosting artifacts.
        if(wt_nxt > 0.3f) wt_nxt = 0.3f; 

        if(wt_prv < 0.001f) wt_prv = 0.0f; // low floating-point clamping
        if(wt_nxt < 0.001f) wt_nxt = 0.0f; // low floating-point clamping

        //adaptive combined values
        adp_val_in = wt_prv*prev_val_in + wt_nxt*nxt_val_in + (1.0f - (wt_prv + wt_nxt))*crnt_val_in;

        //local-backward and local-forward consistency
        adp_val_pr = wt_prv*prev_val_pr + wt_nxt*nxt_val_pr + (1.0f - (wt_prv + wt_nxt))*crnt_val_pr;
        adp_val_pr = wt_prv*last_stab_val + (1.0f - wt_prv)*adp_val_pr;

        adapCmbIn.write(ix, iy, c, adp_val_in);
        adapCmbPr.write(ix, iy, c, adp_val_pr);
    }
}

__global__ void kernel_consist_wt(GPUImageWrapper crntIn, GPUImageWrapper adapCmbIn, GPUImageWrapper consisWt, float beta, float gamma){
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    if (ix >= crntIn.width || iy >= crntIn.height || iy < 0 || ix < 0) {
        return;
    }

    for (int c = 0; c < 3; c++) {
        float crnt_val    = crntIn.read(ix, iy, c);
        float adp_cmb_val = adapCmbIn.read(ix, iy, c);

        // Calculate the weight value
        float wt_val = gamma * expf(-beta * (crnt_val - adp_cmb_val) * (crnt_val - adp_cmb_val));

        // Clamp very small values to zero to avoid precision issues
        if(wt_val < 0.001f) {
            wt_val = 0.0f;
        }

        consisWt.write(ix, iy, c, wt_val);
    }

}

__global__ void kernel_consist_out (GPUImageWrapper consisOut, GPUImageWrapper prevConsis, GPUImageWrapper prevUpdt, GPUImageWrapper crntPr, GPUImageWrapper prevStabWarp,
                                    GPUImageWrapper consWt, float stepSize, float momFac, int isMom)
{
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    if (ix >= consisOut.width || iy >= consisOut.height || iy < 0 || ix < 0) {
        return;
    }

    float lap_pr, lap_out, grad_val, out_val, tmp_1, tmp_2, tmp_3;
    float wt_val;

    for (int c = 0; c < 3; c++) {
        int cnt = 0;

        lap_pr  = 0.0f;
        lap_out = 0.0f;

        //calculating laplacian of per-frame processed result and the consistent output
        if ((ix + 1) < (crntPr.width - 1)) {
            lap_pr += crntPr.read((ix + 1), iy, c);
            lap_out += prevConsis.read((ix + 1), iy, c);
            cnt += 1;
        }

        if ((ix - 1) >= 0) {
            lap_pr += crntPr.read((ix - 1), iy, c);
            lap_out += prevConsis.read((ix - 1), iy, c);
            cnt += 1;
        }

        if ((iy + 1) < (crntPr.height - 1)) {
            lap_pr += crntPr.read(ix, iy + 1, c);
            lap_out += prevConsis.read(ix, iy + 1, c);
            cnt += 1;
        }
        if ((iy - 1) >= 0) {
            lap_pr += crntPr.read(ix, iy - 1, c);
            lap_out += prevConsis.read(ix, iy - 1, c);
            cnt += 1;
        }

        lap_pr  -= cnt * crntPr.read(ix, iy, c);
        lap_out -= cnt * prevConsis.read(ix, iy, c);

        wt_val = consWt.read(ix, iy, c);

        tmp_1 = wt_val*(prevConsis.read(ix, iy, c) - prevStabWarp.read(ix, iy, c));
        tmp_2 = lap_out - lap_pr;

        grad_val = tmp_2 - tmp_1;

        // momentum, insert later again
        if (isMom) {
            out_val = prevConsis.read(ix, iy, c) + stepSize*grad_val + momFac*prevUpdt.read(ix, iy, c);
        } else {
            out_val = prevConsis.read(ix, iy, c) + stepSize*grad_val;
        }

        prevUpdt.write(ix, iy, c, stepSize*grad_val);

        consisOut.write(ix, iy, c, out_val);
    }

}
//////////////////////////////////////// Funtion Calls ///////////////////////////////////////////////

dim3 getGrid(dim3 n, dim3 block) {
    // take ceiling, otherwise some part might be missing!
    dim3 grid = dim3((n.x + block.x) / block.x, (n.y + block.y) / block.y, 1); // number of threads to launch accordingly
    return grid;
}


void perform_consistency(
        GPUImage& originalCurr,
        GPUImage& processedPrev,
        GPUImage& processedCurr,
        GPUImage& processedNext,
        GPUImage& flowFwd,
        GPUImage& flowBwd,
        GPUImage& stabilizedPrev,
        GPUImage& stabilizedOut) {

    dim3 n(stabilizedOut.width, stabilizedOut.height, 1); // number of items to process
    dim3 block(32, 32, 1); // number of items each thread gets
    dim3 grid = getGrid(n, block);
    kernel_average_frames<<<grid, block>>>(CUDA(processedPrev), CUDA(processedNext), CUDA(stabilizedOut));
    checkError("kernel_average_frames");

    cudaDeviceSynchronize();
}

void get_bilinear(
        GPUImage& input,
        GPUImage& output) {
    dim3 n(output.width, output.height, 1); // number of items to process
    dim3 block(32, 32, 1); // number of items each thread gets
    dim3 grid = getGrid(n, block);
    kernel_bilinear<<<grid, block>>>(CUDA(input), CUDA(output));
    checkError("kernel_bilinear");

    cudaDeviceSynchronize();
}

void get_warp_result(
        GPUImage& input,
        GPUImage& flow, //this flow whould be that which warps the current frame to the given "input"
        GPUImage& inputWarp){

    dim3 n(input.width, input.height, 1); // number of items to process
    dim3 block(32, 32, 1); // number of items each thread gets
    dim3 grid = getGrid(n, block);
    kernel_warp<<<grid, block>>>(CUDA(input), CUDA(flow), CUDA(inputWarp));
    checkError("kernel_warp");

    cudaDeviceSynchronize();
}

void get_adap_comb(
        GPUImage& crntIn,
        GPUImage& crntPr,
        GPUImage& prevWarpIn,
        GPUImage& prevWarpPr,
        GPUImage& nextWarpIn,
        GPUImage& nextWarpPr,
        GPUImage& adapCmbIn,
        GPUImage& adapCmbPr,
        GPUImage& lastStabWarp,
        float alpha){


    dim3 n(crntIn.width, crntIn.height, 1); // number of items to process
    dim3 block(32, 32, 1); // number of items each thread gets
    dim3 grid = getGrid(n, block);
    kernel_adap_comb<<<grid, block>>>(CUDA(crntIn), CUDA(crntPr), CUDA(prevWarpIn), CUDA(prevWarpPr), CUDA(nextWarpIn), CUDA(nextWarpPr),
                                      CUDA(adapCmbIn), CUDA(adapCmbPr), CUDA(lastStabWarp), alpha);
    checkError("kernel_adap_comb");

    cudaDeviceSynchronize();

}

void get_consist_wt(
        GPUImage& adapCmbIn,
        GPUImage& crntIn,
        GPUImage& consisWt,
        float  beta,
        float gamma){

    dim3 n(crntIn.width, crntIn.height, 1); // number of items to process
    dim3 block(32, 32, 1); // number of items each thread gets
    dim3 grid = getGrid(n, block);
    kernel_consist_wt<<<grid, block>>>(CUDA(crntIn), CUDA(adapCmbIn), CUDA(consisWt), beta, gamma);
    checkError("kernel_consist_wt");

    cudaDeviceSynchronize();
}

void get_consist_out(
        GPUImage& crntPr,
        GPUImage& prevStabWarp,
        GPUImage& consWt,
        int numIter,
        float stepSize,
        float momFac,
        GPUImage& consisOut){


    dim3 n(consisOut.width, consisOut.height, 1); // number of items to process
    dim3 block(32, 32, 1); // number of items each thread gets
    dim3 grid = getGrid(n, block);

    GPUImage prevUpdt(crntPr.width, crntPr.height, 3);

    GPUImage prevConsis(consisOut.width, consisOut.height, 3);

    for (int k = 0; k < numIter; k++)
    {
        // Using Stochastic Gradient Descent (SGD) for optimization solving
        kernel_consist_out <<< grid, block >>>(CUDA(consisOut), CUDA(consisOut), CUDA(prevUpdt), CUDA(crntPr), CUDA(prevStabWarp), CUDA(consWt), stepSize, momFac, k);
        checkError("kernel_consist_out");
    }
    cudaDeviceSynchronize();
}

