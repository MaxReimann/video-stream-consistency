// Copyright (C) 2023 Max Reimann (max.reimann@hpi.de)
// licensed under GNU General Public License v2.0
#include <stdio.h>
#include <string>
#include <QImage>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

extern "C" {
#include <libavcodec/avcodec.h>
#include "libavformat/avformat.h"
#include "libavutil/display.h"
}


struct FramePts {
    QImage image;
    int64_t pts;
};

class VideoEncoder {
public:
    //takes in the input video path from which to copy codec params and the output video path
    VideoEncoder(const std::string& inputVideoPath, const std::string& outputVideoPath);
    ~VideoEncoder();
    
    void addImage(const QImage& image, int64_t timestamp);
    void finalize();
    void initialize();

private:
    void worker();
    void encodeImage(const QImage& image, int64_t timestamp); 
    void finalPacket();

    std::string m_inputVideoPath;
    std::string m_outputVideoPath;
    
    AVFormatContext* m_inputFormatCtx;
    AVStream* m_inputVideoStream;
    
    AVFormatContext* m_outputFormatCtx;
     AVCodecContext *  m_outputCodecContext;
    AVStream* m_outputVideoStream;
    
    std::queue<FramePts> images;
    std::thread workerThread;
    std::mutex mtx;
    std::condition_variable cv;
    bool m_isRunning;
    
};