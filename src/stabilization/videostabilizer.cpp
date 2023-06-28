/*  FlowConsistency Video Stabilization

    Copyright (C) 2023 Sumit Shekhar (sumit.shekhar@hpi.de) and  Max Reimann (max.reimann@hpi.de)

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
*/


#include "videostabilizer.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <optional>

#include <cmath>
#include <QDir>
#include <QStringList>
#include <QImage>
#include <QDebug>
#include <QSharedPointer>
#include <QList>
#include <QElapsedTimer>


#include "flowconsistency.cuh"
#include "gpuimage.h"
#include "flowIO.h"
#include "imagehelpers.h"
#include "flowmodel.h"


#define WARMUP_FRAMES k+5


VideoStabilizer::VideoStabilizer(int width, int height, int batchSize, std::optional<QString> modelType, bool computeFlow) 
: width(width), height(height), batchSize(batchSize), modelType(modelType),
    stabilizedFrame(GPUImage(width, height, 3)),
    lastStabilizedFrame(GPUImage(width, height, 3)),
    flowFwd(GPUImage(width, height,  computeFlow ? 3 : 2)),
    flowBwd(GPUImage(width, height,  computeFlow ? 3 : 2)),
    //Creating intermediate images
    prevWarpIn(GPUImage(width, height, 3)),
    prevWarpPr(GPUImage(width, height, 3)),
    nextWarpIn(GPUImage(width, height, 3)),
    nextWarpPr(GPUImage(width, height, 3)),
    lastStabWarp(GPUImage(width, height, 3)),
    consisOut(GPUImage(width, height, 3)),
    consWt(GPUImage(width, height, 3)),
    adapCmbIn(GPUImage(width, height, 3)),
    adapCmbPr(GPUImage(width, height, 3))
{
    int flowImageChannels = computeFlow ? 3 : 2;

    for (int r = 0; r < batchSize; r++)
    {
        QSharedPointer<GPUImage> fwd(new GPUImage(width, height, flowImageChannels));
        QSharedPointer<GPUImage> bwd(new GPUImage(width, height, flowImageChannels));
        flowResultsFwd << fwd;
        flowResultsBwd << bwd;
    }

    qDebug() << "Flow forward parameters:" << flowResultsFwd[0]->width << flowResultsFwd[0]->height << flowResultsFwd[0]->channels;
    int flowDownscaleFactor = 1;

    const char* var_name = "FLOWDOWNSCALE";
    char* var_value = std::getenv(var_name);

    if (var_value) {
        try {
            flowDownscaleFactor = std::stoi(var_value);
            std::cout << "The value of " << var_name << " is " << flowDownscaleFactor << "\n";
        } catch (const std::invalid_argument&) {
            std::cout << "Error: " << var_name << " is not an integer.\n";
        } catch (const std::out_of_range&) {
            std::cout << "Error: " << var_name << " is out of range for an integer.\n";
        }
    } else {
        std::cout << "Info: not downsampling flow. Set FLOWDOWNSCALE=x to set the downsampling factor.\n";
    }
    
    if (computeFlow) {
        flowWidth = static_cast<int>(round(static_cast<float>(width) / static_cast<float>(flowDownscaleFactor)));
        flowHeight = static_cast<int>(round(static_cast<float>(height) / static_cast<float>(flowDownscaleFactor)));

        ORT_CONTEXT = std::make_unique<OrtContext>();
        flowModel = std::make_unique<FlowModel>(*modelType, ORT_CONTEXT.get(), batchSize, flowWidth, flowHeight);
    }

    qDebug() << "Frame size:" << width << height;
    initHyperParams();
}

void VideoStabilizer::initHyperParams()
{
    controlParameters.alpha = 6800.0f; // increasing alpha can help reducing ghosting
    controlParameters.beta = 6800.0f;
    controlParameters.gamma = 2.0f; // higher = increased consistency, can also introduce ghosting for fast moving objects
    controlParameters.pyramidLevels = 2;
    controlParameters.numIter = 150;
    controlParameters.stepSize = 0.15f;
    controlParameters.momFac = 0.15f;

    timeStabilized = 0.0;
    timeOptFlow = 0.0;
    timeLoad = 0.0;

    int pyramidW = width;
    int pyramidH = height;
    for (int i = 0; i < controlParameters.pyramidLevels; i++) {
        pyrPr.push_back(QSharedPointer<GPUImage>(new GPUImage(pyramidW, pyramidH, 3)));
        pyrAdapCmbPr.push_back(QSharedPointer<GPUImage>(new GPUImage(pyramidW, pyramidH, 3)));
        pyrConsWt.push_back(QSharedPointer<GPUImage>(new GPUImage(pyramidW, pyramidH, 3)));
        pyrConsisOut.push_back(QSharedPointer<GPUImage>(new GPUImage(pyramidW, pyramidH, 3)));

        pyramidW = pyramidW / 2;
        pyramidH = pyramidH / 2;
    }
}

QString VideoStabilizer::formatIndex(int index) {
    return QString("%1").arg(index, 6, 10, QChar('0'));
};


void VideoStabilizer::preloadProcessedFrames()
{
    for (int j = 0; j < 2*k+batchSize; j++) {
        qDebug() << "preload" << j;
        bool success = loadFrame(j); 
        if (!success){
            throw std::runtime_error("Failed to load initial frames from video!");
        }
        
        // write first k processed frames to output dir
        if (j <= k) {
            auto output = QSharedPointer<QImage>(new QImage(gpuToImage(*processedFrames[j])));
            outputFrame(j, output);
        }
    }

    lastStabilizedFrame.copyFrom(*processedFrames.back());
}

void VideoStabilizer::outputFinalFrames(int currentFrame) {
    qDebug() << "Final frame output";
    // that was the last frame we could process
    // write remaining k processed frames to output dir
    for (int j = 0; j < k; j++) {
        auto res = gpuToImage(*processedFrames[k+j]);
        auto output = QSharedPointer<QImage>(new QImage(res));
        outputFrame(currentFrame+1+j, output);
    }
}


bool VideoStabilizer::doOneStep(int currentFrame) {
    // currentFrame is the frame to process now
    // k frames before, frame currentFrame, k frames after are supposed to be in original/processedFrames list
    // e.g. originalFrames and processedFrames contain 2*k+1 frames, currentFrame here is frame k in these lists
    Q_ASSERT(originalFrames.size() == 2*k+batchSize);
    Q_ASSERT(processedFrames.size() == 2*k+batchSize);

    retrieveOpticalFlow(currentFrame);
    auto beforeWarp = timer.elapsed();

    auto batchIdx = (currentFrame-k) % batchSize;
    flowFwd.copyFrom(*flowResultsFwd[batchIdx].get());
    flowBwd.copyFrom(*flowResultsBwd[batchIdx].get());

    //warp the previous input-frame and processed-frame to the current one
    get_warp_result(*originalFrames[0],flowBwd, prevWarpIn); //warp previous input frame to the current
    get_warp_result(*processedFrames[0],flowBwd, prevWarpPr); //warp previous processed frame to the current

    //warp the next input-frame and processed-frame to the current one
    get_warp_result(*originalFrames[2],flowFwd, nextWarpIn); //warp next input frame to the current
    get_warp_result(*processedFrames[2],flowFwd, nextWarpPr); //warp next processed frame to the current

    //warp previous stabilized frame to the current
    get_warp_result(lastStabilizedFrame, flowBwd, lastStabWarp);

    auto c = controlParameters;
    //combining warped versions to get the otimization intializer
    get_adap_comb(*originalFrames[1], *processedFrames[1], prevWarpIn, prevWarpPr, 
                nextWarpIn, nextWarpPr, adapCmbIn, adapCmbPr, lastStabWarp, c.alpha);

    //computing consistency weights
    get_consist_wt(adapCmbIn, *originalFrames[1], consWt, c.beta, c.gamma);

    bool multiscale = true;
    if (multiscale) {
        // multi-scale solving
        for (int j = 0; j < c.pyramidLevels; j++) {
            if (j == 0) {
                pyrPr[0]->copyFrom(*processedFrames[1]);
                pyrAdapCmbPr[0]->copyFrom(adapCmbPr);
                pyrConsWt[0]->copyFrom(consWt);
                // initialize solution with the per-frame processed result
                pyrConsisOut[0]->copyFrom(*processedFrames[1]);
            } else {
                // downscale input -> output
                get_bilinear(*pyrPr[j-1], *pyrPr[j]);
                get_bilinear(*pyrAdapCmbPr[j-1], *pyrAdapCmbPr[j]);
                get_bilinear(*pyrConsWt[j-1], *pyrConsWt[j]);
                get_bilinear(*pyrConsisOut[j-1], *pyrConsisOut[j]);
            }
        }

        for (int j = c.pyramidLevels-1; j >= 0; j--) {
            // get result from last level
            if (j != c.pyramidLevels-1) {
                // upscale input -> output
                get_bilinear(*pyrConsisOut[j+1], *pyrConsisOut[j]);
            }
            // c.numIter / (j+1) is the number of iterations for the current level of the pyramid, division by j for speedup
            get_consist_out(*pyrPr[j], *pyrAdapCmbPr[j], *pyrConsWt[j], c.numIter / (j+1), c.stepSize, c.momFac, *pyrConsisOut[j]);
        }
        consisOut.copyFrom(*pyrConsisOut[0]);
    } else {
        // single-scale solving
        // initialize solution with the per-frame processed result. 
        consisOut.copyFrom(*processedFrames[1]);
        //solving the optimization to obtain the consistent result for the current frame
        get_consist_out(*processedFrames[1], adapCmbPr, consWt, c.numIter, c.stepSize, c.momFac, consisOut);
    }

    auto out = QSharedPointer<QImage>(new QImage(QSize(consisOut.width, consisOut.height), QImage::Format_RGBA8888));
    consisOut.copyToQImage(*out);

    if (currentFrame > WARMUP_FRAMES)
        timeStabilized += timer.elapsed() - beforeWarp;

    auto beforeSave = timer.elapsed();
    outputFrame(currentFrame, out); 
    auto afterSave = timer.elapsed();

    lastStabilizedFrame.copyFrom(consisOut);
    originalFrames.pop_front();
    originalFramesQt.pop_front();
    processedFrames.pop_front();

    auto beforeLoad = timer.elapsed();
    // i+k is latest frame currently in the list, attempt to load one more
    bool loaded = loadFrame(currentFrame+k+1);
    if (!loaded) {
        outputFinalFrames(currentFrame);
        return false;
    }
    if (currentFrame > FRAME_TIMER_START) {
        timeLoad += timer.elapsed() - beforeLoad;
        timeSave += afterSave - beforeSave;
    }

    return true;
}

void VideoStabilizer::retrieveOpticalFlow(int currentFrame) {
    // Compute batch of optical flow (ever batchsize iterations) and store in flowResults(Fwd/Bwd)
    if ((currentFrame-k) % batchSize == 0) {
        flowTiming timing{0,0}; 
        flowModel->run(originalFramesQt, flowResultsFwd,  1, 2, &timing);
        flowModel->run(originalFramesQt, flowResultsBwd,  2, 1, &timing);

        if (currentFrame > WARMUP_FRAMES) { // measure after warmup
            timeOptFlow += timing.runTime;
            timeLoad += timing.loadTime;
        }
    }
}
