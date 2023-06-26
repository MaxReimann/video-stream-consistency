/*  A simple CUDA-based image representation

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

#include "gpuimage.h"

#include <cuda.h>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <QImage>

void char_to_cuda_float_image(unsigned char* inputImage, size_t num_bytes, GPUImage& output);
void cuda_float_to_char_image(const GPUImage& inputImage,  unsigned char *outputPtr);
void cuda_float_to_char_ptr(const GPUImage& inputImage,  unsigned char **outputPtr);


GPUImage::GPUImage(int width, int height, int channels)
    : width(width), height(height), channels(channels) {
    // allocate data
    size_t nelems = width*height*channels;
    // Allocate space for device tmp buffers
    auto error = cudaMalloc(&data, (unsigned long long) nelems * sizeof(float));
    if (error != cudaError::cudaSuccess) {
        throw std::runtime_error("Unable to allocate CUDA memory.");
    }
}

GPUImage::GPUImage(const GPUImage& other)
    : GPUImage(other.width, other.height, other.channels) {
    copyFrom(other);
}

GPUImage::~GPUImage() {
    // delete data
    cudaFree(data);
}

void GPUImage::copyFrom(const GPUImage& other) {
    if (other.width == width && other.height == height && other.channels == channels) {
        size_t nelems = width*height*channels;
        //std::copy(other.data, other.data + nelems, data);
        const auto errorMemcpy = cudaMemcpy(static_cast<void*>(data), static_cast<void*>(other.data), 
            (unsigned long long) nelems * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        if (errorMemcpy != cudaError::cudaSuccess) {
            throw std::runtime_error("Unable to copy data from device to device.");
        }
    } else {
        std::cerr << "Invalid dimensions" << std::endl;
    }
}

void GPUImage::copyFrom(const std::vector<float>& vec) {
    size_t nelems = width*height*channels;
    if (nelems == vec.size()) {
        //std::copy(vec.begin(), vec.end(), data);
        const auto errorMemcpy = cudaMemcpy(static_cast<void*>(data), &vec.front(),
            (unsigned long long) nelems * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
        if (errorMemcpy != cudaError::cudaSuccess) {
            throw std::runtime_error("Unable to copy data from device to host.");
        }
    } else {
        std::cerr << "Invalid dimensions" << std::endl;
    }
}

void GPUImage::copyFrom(const std::vector<std::byte>& vec) {
    size_t nelems = width*height*channels;
    if (nelems * sizeof(float) == vec.size()) {
        const auto errorMemcpy = cudaMemcpy(static_cast<void*>(data), &vec.front(),
            (unsigned long long) nelems * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
    } else {
        throw std::runtime_error("Invalid dimensions. Length of byte-array differs from expected number of floats");
    }
}


void GPUImage::copyFromCudaBuffer(const void* resourcePointer, size_t byteSize) {
    size_t nelems = width*height*channels;
    if (nelems * sizeof(float) == byteSize) {
        const auto errorMemcpy = cudaMemcpy(static_cast<void*>(data), resourcePointer, byteSize, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        if (errorMemcpy != cudaError::cudaSuccess) {
            throw std::runtime_error("Unable to copy data from device to host.");
        }
    } else {
        throw std::runtime_error("Invalid dimensions. Length of byte-array differs from expected number of floats");
    }
}


void GPUImage::copyFromQImage(const QImage &image) {

    if (image.width() != width || image.height() != height) {
        std::cerr << "Invalid dimensions" << std::endl;
        return;
    }
    
    if (channels != 3) {
        std::cerr << "copyFromQImage requires a 3 channel GPU image" << std::endl;
        return;
    }

    size_t nelems = width*height*4; //  use 4 channels here to have nicely aligned data
    unsigned char *image_data;
    if (image.format() != QImage::Format_RGBA8888) {
        auto tmpRGBA = image.convertToFormat(QImage::Format_RGBA8888);
        image_data = tmpRGBA.bits();
    } else {
        image_data = const_cast<unsigned char *>(image.bits());
    }
    char_to_cuda_float_image(image_data, nelems, *this);
}

// will multiply pixel data by 255 and convert from float to uchar
void GPUImage::copyToQImage(QImage &image) const {
    if (image.format() != QImage::Format_RGBA8888) {
        std::cerr << "copyToQImage needs to be in format RGBA8888" << std::endl;
        return;
    }

    unsigned char *image_data = image.bits();
    cuda_float_to_char_image(*this,  image_data);
}