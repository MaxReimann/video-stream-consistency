#ifndef GPUIMAGEWRAPPER_H
#define GPUIMAGEWRAPPER_H

#include "gpuimage.h"

#include <iostream>

struct GPUImageWrapper {
    float* data;
    int width, height, channels;

    GPUImageWrapper(const GPUImage& image)
        : data(image.data), width(image.width), height(image.height), channels(image.channels) {
        // enable where supported (not windows)
        // cudaMemPrefetchAsync(image.data, width * height * channels * sizeof(float), 0);
    }

    __host__ __device__ float read(int x, int y, int c) {
        return data[(y*width + x) * channels + c];
    }

    __host__ __device__ void write(int x, int y, int c, float value) {
        data[(y*width + x) * channels + c] = value;
    }
};

#endif // GPUIMAGEWRAPPER_H
