/*  Video Stream-based stabilization

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

#include "stabilizestream.h"
#include <QDir>
#include <QCoreApplication>
#include <optional>
#include <thread>

#define WARMUP_FRAMES (k+5)

StreamStabilizer::StreamStabilizer(QDir originalVidPath,  QDir processedVidPath, std::optional<QString> stabilizedDir, std::optional<QString> modelType, int width, int height, int batchSize, bool streamingOutput) : 
VideoStabilizer(width, height, batchSize,  modelType, true), originalVidPath(originalVidPath), processedVidPath(processedVidPath), stabilizedDirorVid(stabilizedDir), streamingOutput(streamingOutput),
videoDecode(TwoStreamDecoder(0.0, originalVidPath.absolutePath().toStdString(), processedVidPath.absolutePath().toStdString())),
flowVis(GPUImage(width, height, 3))
 {
    qDebug() << originalVidPath.absolutePath();
    if ( !(QFileInfo::exists(originalVidPath.absolutePath()) && !originalVidPath.exists()) ){
        throw std::runtime_error("Original vid does not exist!");
    }
    if ( !(QFileInfo::exists(processedVidPath.absolutePath()) && !processedVidPath.exists()) ){
        throw std::runtime_error("Processed vid does not exist!");
    }

    if (!QFileInfo(*stabilizedDirorVid).isDir()) {
       videoEncoder = std::make_unique<VideoEncoder>(originalVidPath.absolutePath().toStdString(), stabilizedDirorVid->toStdString());
    }
}


bool StreamStabilizer::loadFrame(int i)
{
    std::unique_ptr<CombinedFrame> frame_info{nullptr};
    // pop waits for the next frame to be available, returns false if queue is finished and empty, or it is quit
    while (!videoDecode.get_frame_queue().pop(frame_info)) 
    {
        if (getVideoControl()->get_quit())
        {
            videoDecode.get_frame_queue().quit();
            std::cout << "[StreamStabilizer::loadFrame] Frame queue quit" << std::endl;
            return false;
        } 

        if (videoDecode.get_frame_queue().is_finished())
        {
            videoDecode.get_frame_queue().quit();
            std::cout << "[StreamStabilizer::loadFrame] Frame queue finished" << std::endl;
            return false;
        }
    }

    if (frame_info == nullptr)
    {
        std::cout << "[StreamStabilizer::loadFrame] Frame info is null" << std::endl;
        return false;
    }

    m_lastPts = frame_info->pts;
    originalFramesQt << frame_info->original;
    originalFrames << imageToGPU(*frame_info->original.get());
    processedFrames << imageToGPU(*frame_info->processed.get());
    m_last_frame_info = std::move(frame_info);
    return true;
}

QString StreamStabilizer::formatIndex(int index) {
    return QString("%1").arg(index, 6, 10, QChar('0'));
};

void StreamStabilizer::outputFrame(int i, QSharedPointer<QImage> q) {
    m_lastImage = q;
    if (streamingOutput) {
        switch (videoDecode.get_video_control()->get_video_type()) {
            case VideoType::STYLIZED:
                emit frameReady(m_lastPts, m_last_frame_info->processed);
                break;
            case VideoType::ORIGINAL:
                emit frameReady(m_lastPts, m_last_frame_info->original);
                break;
            case VideoType::FLOWVIS:
                emit frameReady(m_lastPts, computeFlowVis(i));
                break;
            default:
                // We use the last pts instead of frame num to enable seeking in the output video
                emit frameReady(m_lastPts, q);
                break;
        }
    } else {
        // check if stabilizedDir is a dir or a video using qfileinfo
        if (!QFileInfo(*stabilizedDirorVid).isDir()) {
            videoEncoder->addImage(*q, m_lastPts);
        } else {
            q->save(QDir(*stabilizedDirorVid).filePath(formatIndex(i) + ".jpg"));
        }

    }
}

void StreamStabilizer::startBackgroundThread() {
    videoDecodeThread = std::thread(&TwoStreamDecoder::run, &videoDecode); // start video decoding in background
    preloadProcessedFrames();
}

bool StreamStabilizer::processOneFrame(int i) {
    if (videoDecode.get_exception()){
        std::rethrow_exception(videoDecode.get_exception());
    }

    bool success = doOneStep(i); 
    if (!success) {
        std::cout << "[StreamStabilizer::processOneFrame] Finished processing at frame " << i << std::endl;
        videoDecode.get_video_control()->set_quit(true);
        return false;
    }

    if (i == 100 + WARMUP_FRAMES) {
        int count = i - WARMUP_FRAMES;
        qDebug() << "Per-frame time in ms averaged over 100 frames (without first " << WARMUP_FRAMES << " for warmup):" << "load (wait+to_gpu): " << 
            timeLoad / float(count) << "optflow: " << timeOptFlow  / float(count)  <<
            "save: " << timeSave / float(count) << 
            "stabilize: " <<  timeStabilized / float(count) << 
            "total: " << (timeLoad + timeOptFlow + timeSave + timeStabilized) / float(count);
    }

    return true;
}


bool StreamStabilizer::stabilizeAll() {
    startBackgroundThread();
    timer.start();
    for (int i = k;; i++) {
        bool success = processOneFrame(i);    
        if (!success) {
            break;
        }
    }

    videoDecodeThread.join();

    return true;
} 

void StreamStabilizer::quit() {
    videoDecode.get_video_control()->set_quit(true);
    if (videoEncoder)
        videoEncoder->finalize();
    videoDecodeThread.join();
}

QSharedPointer<QImage> StreamStabilizer::computeFlowVis(int currentFrame)
{   
    QList<QSharedPointer<GPUImage>> flowvisResults;
    flowTiming timing{0,0}; 
    QSharedPointer<GPUImage> flowVisP = QSharedPointer<GPUImage>::create(flowVis);
    flowvisResults.push_back(flowVisP);
    flowModel->runFlowVis(flowvisResults, &timing);

    QSharedPointer<QImage> flowVisQ(new QImage(flowVis.width, flowVis.height, QImage::Format_RGBA8888));
    flowvisResults[0]->copyToQImage(*flowVisQ);
    timeOptFlow += timing.runTime;
    return flowVisQ;
}
