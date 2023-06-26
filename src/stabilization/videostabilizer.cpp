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

    qDebug() << "flow fwd" << flowResultsFwd[0]->width << flowResultsFwd[0]->height << flowResultsFwd[0]->channels;
    int flowDownscaleFactor = 1;

    const char* var_name = "FLOWDOWNSCALE";
    char* var_value = std::getenv(var_name);

    if (var_value != NULL) {
        try {
            int value = std::stoi(var_value);
            std::cout << "The value of " << var_name << " is " << value << "\n";
            flowDownscaleFactor = value;
        } catch (std::invalid_argument& e) {
            std::cout << "Error: " << var_name << " is not an integer.\n";
        } catch (std::out_of_range& e) {
            std::cout << "Error: " << var_name << " is out of range for an integer.\n";
        }
    } else
    {
        std::cout << "Info: not downsampling flow. Set FLOWDOWNSCALE=x to set the downsampling factor.\n";
    }
    

    
    // QSharedPointer<GPUImage> flowLowRes;
    if (computeFlow) {
        flowWidth = static_cast<int>(round(static_cast<float>(width) / static_cast<float>(flowDownscaleFactor)));
        flowHeight = static_cast<int>(round(static_cast<float>(height) / static_cast<float>(flowDownscaleFactor)));

        ORT_CONTEXT = std::make_unique<OrtContext>();
        flowModel = std::make_unique<FlowModel>(*modelType, ORT_CONTEXT.get(), batchSize, flowWidth, flowHeight);
    }

    qDebug() << "Frame size:" << width << height;
    // qDebug() << "Frame count:" << frameCount;

    initHyperParams();

}



void VideoStabilizer::initHyperParams()
{
    controlParameters.alpha = 6800.0f; // increasing alpha and/or gamma to fix ghosting artifacts does not work!
    controlParameters.beta = 6800.0f; // when we increase it too much I see some problems... maybe due to controlParameters.ng point overflow or something similar
    controlParameters.gamma = 2.0f;//3.0;//1.9f; // make it high to increase consistency. however to remove ghosting artiffacts due to fast motion... reduce it to around 0.1
    //controlParameters.gamma = 0.1f; //such low gamma value helps to avoid the ghosting artifacts.... but then consistency is also very low!!!
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
        // for (int j = 0; j <= k; j++) {
        if (j <= k) {
            // (processedFrames contains 2k+1 frames, k before current and k after...)
            auto output = QSharedPointer<QImage>(new QImage(gpuToImage(*processedFrames[j])));
            outputFrame(j, output);
        }
    }

    qDebug() << "processedFrames length" << processedFrames.length();
    lastStabilizedFrame.copyFrom(*processedFrames.back());
}

void VideoStabilizer::outputFinalFrames(int currentFrame) {
    qDebug() << "stop";
    // that was the last frame we could process
    // write remaining k processed frames to output dir
    for (int j = 0; j < k; j++) {
        // (processedFrames contains only 2k frames at this point!! k-1 before current, 1 current, k after...)
        // gpuToImage(*processedFrames[k+j]).save(stabilizedDir.filePath(formatIndex(i+1+j) + ".jpg"));
        auto res = gpuToImage(*processedFrames[k+j]);
        auto output = QSharedPointer<QImage>(new QImage(res));
        outputFrame(currentFrame+1+j, output);
    }
}


bool VideoStabilizer::doOneStep(int currentFrame) {
    // qDebug() << "step" << currentFrame;
    // currentFrame is the frame to process now
    // k frames before, frame currentFrame, k frames after are supposed to be in original/processedFrames list
    // e.g. originalFrames and processedFrames contain 2*k+1 frames, currentFrame here is frame k in these lists

    auto c = controlParameters;
    // do processing step
    Q_ASSERT(originalFrames.size() == 2*k+batchSize);
    Q_ASSERT(processedFrames.size() == 2*k+batchSize);

    // qDebug() << "frame_" + formatIndex(i + 1) + ".flo";
    retrieveOpticalFlow(currentFrame);

    auto beforeWarp = timer.elapsed();

    auto batchIdx = (currentFrame-k) % batchSize;
    flowFwd.copyFrom(*flowResultsFwd[batchIdx].get());
    flowBwd.copyFrom(*flowResultsBwd[batchIdx].get());


    //warp the previous input-frame and processed-frame to the current one
    get_warp_result(*originalFrames[0],flowBwd, prevWarpIn); //warp previous input frame to the current
    get_warp_result(*processedFrames[0],flowBwd, prevWarpPr); //warp previous processed frame to the current
    //qDebug() << timer.elapsed() << "warp fwd";

    //warp the next input-frame and processed-frame to the current one
    get_warp_result(*originalFrames[2],flowFwd, nextWarpIn); //warp next input frame to the current
    get_warp_result(*processedFrames[2],flowFwd, nextWarpPr); //warp next processed frame to the current
    //qDebug() << timer.elapsed() << "warp bwd";

    //warp previous stabilized frame to the current
    get_warp_result(lastStabilizedFrame, flowBwd, lastStabWarp);
    //qDebug() << timer.elapsed() << "warp stabilized";

    //combining warped versions to get the otimization intializer
    get_adap_comb(*originalFrames[1], *processedFrames[1], prevWarpIn, prevWarpPr, 
                nextWarpIn, nextWarpPr, adapCmbIn, adapCmbPr, lastStabWarp, c.alpha);
    //qDebug() << timer.elapsed() << "get_adap_comb";

    //computing consistency weights
    get_consist_wt(adapCmbIn, *originalFrames[1], consWt, c.beta, c.gamma);
    //qDebug() << timer.elapsed() << "get_consist_wt";

    bool multiscale = true;
    if (multiscale) {
        // multi-scale solving
        for (int j = 0; j < c.pyramidLevels; j++) {
            if (j == 0) {
                pyrPr[0]->copyFrom(*processedFrames[1]);
                pyrAdapCmbPr[0]->copyFrom(adapCmbPr);
                pyrConsWt[0]->copyFrom(consWt);
                // initialize solution with the per-frame processed result
                //pyrConsisOut[0]->copyFrom(adapCmbPr);
                pyrConsisOut[0]->copyFrom(*processedFrames[1]);
            } else {
                // downscale input -> output
                //Q_ASSERT(pyrPr[j-1]->width == pyrPr[j]->width * 2);
                //Q_ASSERT(pyrPr[j-1]->height == pyrPr[j]->height * 2);
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
                //Q_ASSERT(pyrConsisOut[j+1]->width * 2 == pyrConsisOut[j]->width);
                //Q_ASSERT(pyrConsisOut[j+1]->height * 2 == pyrConsisOut[j]->height);
                //qDebug() << j+1 << "->" << j;
                get_bilinear(*pyrConsisOut[j+1], *pyrConsisOut[j]);
            }
            // c.numIter / (j+1) is the number of iterations for the current level of the pyramid, division by j for speedup
            get_consist_out(*pyrPr[j], *pyrAdapCmbPr[j], *pyrConsWt[j], c.numIter / (j+1), c.stepSize, c.momFac, *pyrConsisOut[j]);
        }
        consisOut.copyFrom(*pyrConsisOut[0]);
    } else {
        // single-scale solving
        // initialize solution with the per-frame processed result. Initiailzing it with adapCmb leads to ghosting artifacts
        //consisOut.copyFrom(adapCmbPr);
        consisOut.copyFrom(*processedFrames[1]);
        //solving the optimization to obtain the consistent result for the current frame
        get_consist_out(*processedFrames[1], adapCmbPr, consWt, c.numIter, c.stepSize, c.momFac, consisOut);
        //qDebug() << timer.elapsed() << "get_consist_out";
    }

    // qDebug() << timer.elapsed() << "wrote result";
    auto out = QSharedPointer<QImage>(new QImage(QSize(consisOut.width, consisOut.height), QImage::Format_RGBA8888));
    consisOut.copyToQImage(*out);

    if (currentFrame > k + 5)
        timeStabilized += timer.elapsed() - beforeWarp;

    auto beforeSave = timer.elapsed();
    outputFrame(currentFrame, out); // output->save(stabilizedDir.filePath(formatIndex(i) + ".jpg"));
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
    if (currentFrame > k + 5) {
        timeLoad += timer.elapsed() - beforeLoad;
        timeSave += afterSave - beforeSave;
    }
    // qDebug() << timer.elapsed() << "loaded next image";

    return true;
}

void VideoStabilizer::retrieveOpticalFlow(int currentFrame) {
    // Compute batch of optical flow (ever batchsize iterations) and store in flowResults(Fwd/Bwd)
    if ((currentFrame-k) % batchSize == 0) {
        flowTiming timing{0,0}; 
        flowModel->run(originalFramesQt, flowResultsFwd,  1, 2, &timing);
        flowModel->run(originalFramesQt, flowResultsBwd,  2, 1, &timing);

        if (currentFrame > k + 5) { // measure after warmup
            timeOptFlow += timing.runTime;
            timeLoad += timing.loadTime;
        }
    }
}