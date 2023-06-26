/*  File-based stabilization

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

#include "stabilizefiles.h"

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

FileStabilizer::FileStabilizer(QDir originalFrameDir, QDir processedFrameDir, QDir stabilizedFrameDir, std::optional<QDir> opticalFlowDir,  std::optional<QString> modelType, int width, int height, int batchSize, bool computeOpticalFlow) : VideoStabilizer(width, height, batchSize,  modelType, computeOpticalFlow), originalFrameDir(originalFrameDir), processedFrameDir(processedFrameDir), stabilizedFrameDir(stabilizedFrameDir), opticalFlowDir(opticalFlowDir)
{
    qDebug() << originalFrameDir.absolutePath();
    if (!originalFrameDir.exists()) {
        throw std::runtime_error("Original frame dir does not exist!");
    }
    if (!processedFrameDir.exists()) {
        throw std::runtime_error("Processed frame dir does not exist!");
    }

    if (opticalFlowDir && !opticalFlowDir->exists()) {
        throw std::runtime_error("Optical flow dir does not exist!");
    }


    // gather input files
    QStringList originalFramePaths, processedFramePaths;
    for (QString name : originalFrameDir.entryList(QStringList({"*.png", "*.jpg"}), QDir::Files, QDir::Name)) {
        originalFramePaths << originalFrameDir.filePath(name);
    }
    for (QString name : processedFrameDir.entryList(QStringList({"*.png", "*.jpg"}), QDir::Files, QDir::Name)) {
        processedFramePaths << processedFrameDir.filePath(name);
    }
    frameCount = std::min(originalFramePaths.size(), processedFramePaths.size());
    Q_ASSERT(frameCount <= originalFramePaths.size());
    Q_ASSERT(frameCount <= processedFramePaths.size());

    inputPaths = std::make_optional(pathInputs{originalFramePaths,processedFramePaths, frameCount});
    
    stabilizedFrameDir.mkpath(".");

}

bool FileStabilizer::loadFrame(int i)
{
    QSharedPointer<QImage> image(new QImage(QImage(inputPaths->originalFramePaths[i]).scaled(width, height)));
    originalFramesQt << image;
    originalFrames << imageToGPU(*image.get());
    QImage imageProcessed;
    imageProcessed.load(inputPaths->processedFramePaths[i]);
    processedFrames << imageToGPU(imageProcessed.scaled(width, height));
    return true;
}


QString FileStabilizer::formatIndex(int index) {
    return QString("%1").arg(index, 6, 10, QChar('0'));
};


void FileStabilizer::outputFrame(int i, QSharedPointer<QImage> q) {
    q->save(stabilizedFrameDir.filePath(formatIndex(i) + ".jpg"));
}

bool FileStabilizer::stabilizeAll() {
    preloadProcessedFrames();
    for (int i = k;; i++) {
        // i is the frame to process now
        // k frames before, frame i, k frames after are supposed to be in original/processedFrames list
        // e.g. originalFrames and processedFrames contain 2*k+1 frames, current frame i here is frame k in these lists
        timer.start();
        bool success = doOneStep(i); 
        if (!success) {
            break;
        }

        if (i == 105) {
            int count = i - (k+5);
            qDebug() << "Per-frame time in ms averaged over 100 frames (without first 5 for warmup):" << "load (wait+to_gpu): " << 
                timeLoad / float(count) << "optflow: " << timeOptFlow  / float(count)  <<
                "save: " << timeSave / float(count) << 
                "stabilize: " <<  timeStabilized / float(count) << 
                "overall: " << (timeLoad + timeOptFlow + timeStabilized + timeSave) / float(count);
            // videoDecode.get_video_control()->set_quit(true);
            // break;
        }
    }
    return true;
}


void FileStabilizer::retrieveOpticalFlow(int currentFrame)
{
    if (opticalFlowDir) {
        std::vector<float> flowTmp;
        int flowWidth, flowHeight;
        // flow file i descibes optical flow i-1 -> i     (last -> current)
        // bwd flow file describes optical flow i -> i-1  (current -> last)
        ReadFlowFile(flowTmp, flowWidth, flowHeight, opticalFlowDir->filePath("frame_" + formatIndex(currentFrame + 1) + ".flo").toStdString().c_str());
        initializeFlowImage(flowTmp, flowFwd, flowWidth, flowHeight, width, height);

        ReadFlowFile(flowTmp, flowWidth, flowHeight, opticalFlowDir->filePath("frame_" + formatIndex(currentFrame) + "_bwd.flo").toStdString().c_str()); 
        initializeFlowImage(flowTmp, flowBwd, flowWidth, flowHeight, width, height);
    } else {
        VideoStabilizer::retrieveOpticalFlow(currentFrame);
    }
}