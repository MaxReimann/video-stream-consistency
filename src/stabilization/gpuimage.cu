/*  CUDA kernels for GPUImage and conversion

    Copyright (C) 2020 Moritz Hilscher
    Modified  2023 Max Reimann (max.reimann@hpi.de)

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
*/

#include "gpuimagewrapper.h"
#include "gpuimage.h"

#include <algorithm>

#define CUDA(image) GPUImageWrapper((image))

dim3 getGrid_(dim3 n, dim3 block) {
    // take ceiling, otherwise some part might be missing!
    dim3 grid = dim3((n.x + block.x) / block.x, (n.y + block.y) / block.y, 1); // number of threads to launch accordingly
    return grid;
}

void checkError_(std::string stage) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error at " << stage << ": " << error << ", " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

// in array is assumed to be an image in the format of rgba8888, the alpha channel isn't used
__global__ void kernel_to_float_image(unsigned char* in,  GPUImageWrapper out) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    if (ix >= out.width || iy >= out.height || iy < 0 || ix < 0) {
        return;
    }

    for (int c = 0; c < 3; c++) {
        unsigned char v1 = in[(iy*out.width + ix) * 4 + c];
        out.write(ix, iy, c, static_cast<float>(v1) / 255.0 );
    }
}

// out array is assumed to be an image in the format of rgba8888
__global__ void kernel_to_char_image(GPUImageWrapper in, unsigned char* out) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    if (ix >= in.width || iy >= in.height || iy < 0 || ix < 0) {
        return;
    }

    for (int c = 0; c < 3; c++) {
        float value = in.read(ix, iy, c);
        out[(iy*in.width + ix) * 4 + c] = static_cast<unsigned char>(__float2uint_rd(fabs(value) * 255));
    }
    out[(iy*in.width + ix) * 4 + 3] = 1;
}


void char_to_cuda_float_image(
        unsigned char* inputImage, size_t num_bytes,
        GPUImage& output){
    
    unsigned char *device_data;  // device tmp buffer
    // Allocate space for device tmp buffers
    auto error = cudaMalloc((void**)&device_data, num_bytes);
    if (error != cudaError::cudaSuccess) {
        throw std::runtime_error("Unable to allocate CUDA memory.");
    }

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    if (num_bytes > free_mem) {
        std::cout << "Free memory: " << free_mem / (1024.0 * 1024.0) << " MB\n";
        std::cout << "Total memory: " << total_mem / (1024.0 * 1024.0) << " MB\n";
        std::cout << "Trying to allocate: " << num_bytes / (1024.0 * 1024.0) << " MB\n";
        throw std::runtime_error("Not enough GPU memory for allocation.");
    }
    const auto errorMemcpy = cudaMemcpy(device_data, inputImage, num_bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (errorMemcpy != cudaError::cudaSuccess) {
        throw std::runtime_error("Unable to copy data from host to device.");
    }


    dim3 n(output.width, output.height, 1); // number of items to process
    dim3 block(32, 32, 1); // number of items each thread gets
    dim3 grid = getGrid_(n, block);
    kernel_to_float_image<<<grid, block>>>(device_data, CUDA(output));
    checkError_("kernel_to_float_image");

    cudaDeviceSynchronize();
    cudaFree(device_data);
}


void cuda_float_to_char_ptr(const GPUImage& inputImage,  unsigned char **output_device_data){
    // Allocate space for device tmp buffers
    size_t num_bytes = inputImage.height * inputImage.width * 4;
    auto error = cudaMalloc((void**)output_device_data, num_bytes);
    if (error != cudaError::cudaSuccess) {
        throw std::runtime_error("Unable to allocate CUDA memory.");
    }

    dim3 n(inputImage.width, inputImage.height, 1); // number of items to process
    dim3 block(32, 32, 1); // number of items each thread gets
    dim3 grid = getGrid_(n, block);
    kernel_to_char_image<<<grid, block>>>(CUDA(inputImage), *output_device_data);
    checkError_("kernel_to_char_image");
}

void cuda_float_to_char_image(const GPUImage& inputImage,  unsigned char *outputPtr){
    
    unsigned char *device_data;  // device tmp buffer
    cuda_float_to_char_ptr(inputImage, &device_data);
    cudaDeviceSynchronize();

    size_t num_bytes = inputImage.height * inputImage.width * 4;
    const auto errorMemcpy = cudaMemcpy(outputPtr, device_data, num_bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    if (errorMemcpy != cudaError::cudaSuccess) {
        throw std::runtime_error("Unable to copy data from device to host.");
    }

    cudaFree(device_data);
}
