#include "imagehelpers.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cstring>

QImage gpuToImage(const GPUImage& gpuImage, bool grayscale) {
    QImage image(QSize(gpuImage.width, gpuImage.height), QImage::Format_RGBA8888);
    int gray = !grayscale;

    gpuImage.copyToQImage(image);
    return image;
}

QSharedPointer<GPUImage> imageToGPU(const QImage& image) {
    QSharedPointer<GPUImage> gpuImage(new GPUImage(image.width(), image.height(), 3));
    gpuImage->copyFromQImage(image);
    return gpuImage;
}

void cpyNImagesToBuffer(QList<QSharedPointer<QImage>>& images, int batchDim, std::vector<std::byte>& outBuffer) {

    if (images.size() < batchDim + 2) {
        std::cerr << "images list must contain at least batchdim +2 images" << std::endl;
        return;
    }

    auto imBytes = images[0]->width() * images[0]->height() * 4;
    outBuffer.reserve(imBytes * (batchDim+2));

    for (int i = 0; i < (batchDim+2); i++)
    {
        auto imOffset = imBytes * i;
        auto converted = images[i]->convertToFormat(QImage::Format_RGBA8888);
        std::byte* data = reinterpret_cast<std::byte*>(converted.bits()) ;

        std::memcpy((void*) &outBuffer[imOffset], (void*) data, imBytes);
    }
}

void initializeFlowImage(std::vector<float>& in_flow_data, GPUImage& out_flow,
    int flowWidth, int flowHeight, int imageWidth, int imageHeight) {

    if (flowWidth == imageWidth && flowHeight == imageHeight) {
        out_flow.copyFrom(in_flow_data);
    } else {
        throw std::runtime_error("Flow image size does not match image size");
    }
}

void initializeFlowImage(void* in_flow_resource_ptr, size_t byte_size, GPUImage& out_flow,
    int flowWidth, int flowHeight, int imageWidth, int imageHeight) {

    if (flowWidth == imageWidth && flowHeight == imageHeight) {
        out_flow.copyFromCudaBuffer(in_flow_resource_ptr, byte_size);
    } else {
        throw std::runtime_error("Flow image size does not match image size");
    }
}
