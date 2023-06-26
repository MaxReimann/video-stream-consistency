#pragma once

#include <QImage>
#include <QSharedPointer>
#include <QList>

#include "gpuimage.h"



QImage gpuToImage(const GPUImage& gpuImage, bool grayscale = false);
QSharedPointer<GPUImage> imageToGPU(const QImage& image);
void cpyNImagesToBuffer(QList<QSharedPointer<QImage>>& images, int batchDim, std::vector<std::byte>& outBuffer);
void initializeFlowImage(std::vector<float>& in_flow_data, GPUImage& out_flow,
    int flowWidth, int flowHeight, int imageWidth, int imageHeight);
void initializeFlowImage(void* in_flow_resource_ptr, size_t byte_size, GPUImage& out_flow,
    int flowWidth, int flowHeight, int imageWidth, int imageHeight);