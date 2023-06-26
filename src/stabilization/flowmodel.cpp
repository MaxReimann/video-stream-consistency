/*  Optical Flow Model

    Copyright (C) 2023 Max Reimann (max.reimann@hpi.de)

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
*/


#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <optional>

#include <cmath>
#include <QDir>
#include <QImage>
#include <QSharedPointer>
#include <QList>
#include <QElapsedTimer>

#include <onnxruntime_cxx_api.h>
#include <inference/InferenceModelVariant.h>
#include <inference/ImageArrayIOHelper.h>
#include <inference/CpuIO.h>
#include <inference/CudaIO.h>
#include <inference/OrtContext.h>


#include "flowmodel.h"
#include "gpuimage.h"
#include "imagehelpers.h"
#include "flowconsistency.cuh"


FlowModel::FlowModel(QString modelChoice,  OrtContext* ort_context, 
        int batchSize, int width, int height)
        : batchSize(batchSize), width(width), height(height), 
        _ort_context(ort_context)
{

    std::string _path;
    if (modelChoice.compare("pwcnet-light") == 0) {
        _path = "models/PWCNet-light-wpreproc.onnx";
    } else if (modelChoice.compare("pwcnet") == 0) {
        _path = "models/PWCNet-dense-wpreproc.onnx";
    } else {
        throw std::runtime_error("Unknown model choice");
    }

    auto modelPath = std::filesystem::path(_path);    
    InferenceProvider provider = InferenceProvider::CUDA;

    switch (provider) {
        case InferenceProvider::CPU:
            outputFlow = ImageArrayIOHelper::createImageArrayIO<CpuIO<float_t>>(width, height, 3, batchSize);
            flowVis = ImageArrayIOHelper::createImageArrayIO<CpuIO<float_t>>(width, height, 3, 0);
            for (int i = 0; i < 2; i++) {
                nmpInputImages.push_back(ImageArrayIOHelper::createImageArrayIO<CpuIO<uint8_t>>(width, height, 4, batchSize));
            }
            break;
        case InferenceProvider::CUDA:
            outputFlow = ImageArrayIOHelper::createImageArrayIO<CudaIO<float_t>>(width, height, 3, batchSize);
            flowVis = ImageArrayIOHelper::createImageArrayIO<CudaIO<float_t>>(width, height, 3, 0);
            for (int i = 0; i < 2; i++) {
                nmpInputImages.push_back(ImageArrayIOHelper::createImageArrayIO<CudaIO<uint8_t>>(width, height, 4, batchSize));
            }
            break;
    }

    InferenceModelVariant::MappedIO input1{"frame1", nmpInputImages[0]};
    InferenceModelVariant::MappedIO input2{"frame2", nmpInputImages[1]};

    std::map<std::string, InferenceModelVariant::MappedIO> inputs; 
    inputs.emplace("frame1", std::move(input1));
    inputs.emplace("frame2", std::move(input2));

    InferenceModelVariant::MappedIO outputFlowMapped{"output", outputFlow};

    std::map<std::string, InferenceModelVariant::MappedIO> outputs;
    outputs.emplace("output", std::move(outputFlowMapped));

    model = std::make_unique<InferenceModelVariant>(modelPath,
        provider,
        *_ort_context,
        std::move(inputs),
        std::move(outputs)
    );

    model->loadSession();

    // create inputs for flowvis from outputFlow and init flowVis model
    InferenceModelVariant::MappedIO flowVisInput{"flow", outputFlow};
    std::map<std::string, InferenceModelVariant::MappedIO> inputsFlowVis;
    inputsFlowVis.emplace("flow", std::move(flowVisInput));

    InferenceModelVariant::MappedIO flowVisOutput{"output", flowVis};
    std::map<std::string, InferenceModelVariant::MappedIO> outputsFlowVis;
    outputsFlowVis.emplace("output", std::move(flowVisOutput));
    
    auto flowVisModelPath = std::filesystem::path("models/flowvis.onnx");
    flowvismodel = std::make_unique<InferenceModelVariant>(flowVisModelPath,
        provider,
        *_ort_context,
        std::move(inputsFlowVis),
        std::move(outputsFlowVis)
    );

    flowvismodel->loadSession();

}

void FlowModel::run(QList<QSharedPointer<QImage>>& originalFramesQt, QList<QSharedPointer<GPUImage>>& results, int indexFirst, int indexSecond,flowTiming* timing) {
    
    auto beforeLoad = timer.elapsed();
    std::vector<std::byte> imgBuffer;

    if (width != originalFramesQt[0]->width() || height != originalFramesQt[0]->height()) {
        // resize images
        QList<QSharedPointer<QImage>> resizedFramesQt;
        for (int i = 0; i < originalFramesQt.size(); i++) {
            auto resized = originalFramesQt[i]->scaled(width, height, Qt::IgnoreAspectRatio, Qt::FastTransformation);
            resizedFramesQt.push_back(QSharedPointer<QImage>::create(resized));
        }
        cpyNImagesToBuffer(resizedFramesQt, batchSize, imgBuffer);
    } else {
    cpyNImagesToBuffer(originalFramesQt, batchSize, imgBuffer);
    }
    
    auto bytesPerImage = static_cast<size_t>(width) * height * 4; 
    std::byte* im1 = &imgBuffer[indexFirst*bytesPerImage];
    std::byte* im2 = &imgBuffer[indexSecond*bytesPerImage];

    nmpInputImages[0]->setData(im1, bytesPerImage * batchSize);
    nmpInputImages[1]->setData(im2, bytesPerImage * batchSize);

    timing->loadTime +=  timer.elapsed() - beforeLoad;
    auto beforerun = timer.elapsed();

    model->run();

    // flow format R32G32B32_FLOAT
    int flowHeight = outputFlow->shape()[1];
    int flowWidth = outputFlow->shape()[2];
    auto byteNum = outputFlow->byteSize() / batchSize;

    // copy output to batchsize number GPUImages
    for (int b=0; b < batchSize; b++)
    {
        if (width != originalFramesQt[0]->width() || height != originalFramesQt[0]->height()) {
            GPUImage tmpFlowLowRes(flowWidth, flowHeight, 3);
            tmpFlowLowRes.copyFromCudaBuffer( reinterpret_cast<std::byte*>(outputFlow->resourcePointer()) + byteNum*b, byteNum);
            get_bilinear(tmpFlowLowRes, *results[b].get());
        } else {
            results[b]->copyFromCudaBuffer( reinterpret_cast<std::byte*>(outputFlow->resourcePointer()) + byteNum*b, byteNum);
        }
    }

    timing->runTime += timer.elapsed() - beforerun;
}

 void FlowModel::runFlowVis(QList<QSharedPointer<GPUImage>> &results, flowTiming *timing) {
    auto beforerun = timer.elapsed();

    flowvismodel->run();
    // flow format R32G32B32_FLOAT (h,w,3) -- the batchchannel is dropped
    int flowHeight = flowVis->shape()[0];
    int flowWidth = flowVis->shape()[1];
    auto byteNum = flowVis->byteSize() / batchSize;

    // copy output to batchsize number GPUImages
    for (int b=0; b < batchSize; b++)
    {
        std::cout << "flowWidth: " << flowWidth << " flowHeight: " << flowHeight << std::endl;
        std::cout << "results[b]->width: " << results[b]->width << " results[b]->height: " << results[b]->height << std::endl;
        if (flowWidth != results[0]->width || flowHeight != results[0]->height) {
            GPUImage tmpFlowLowRes(flowWidth, flowHeight, 3);
            tmpFlowLowRes.copyFromCudaBuffer( reinterpret_cast<std::byte*>(flowVis->resourcePointer()) + byteNum*b, byteNum);
            get_bilinear(tmpFlowLowRes, *results[b].get());
        } else {
            results[b]->copyFromCudaBuffer( reinterpret_cast<std::byte*>(flowVis->resourcePointer()) + byteNum*b, byteNum);
        }
    }


    timing->runTime += timer.elapsed() - beforerun;
 }

