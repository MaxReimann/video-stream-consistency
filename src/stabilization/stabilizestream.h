#pragma once
#include "videostabilizer.h"
#include "../decoding/video_control.h"
#include "../decoding/twostream_decoder.h"
#include "../decoding/encoder.h"

#include <QDir>
#include <optional>
#include <QStringList>
#include <QDebug>


class StreamStabilizer :  public QObject, public VideoStabilizer
{
    Q_OBJECT

    private:
        QDir originalVidPath;
        QDir processedVidPath;
        std::optional<QString> stabilizedDirorVid;
        int frameCount;
        bool streamingOutput;
        std::thread videoDecodeThread;
        TwoStreamDecoder videoDecode;
        int64_t m_lastPts = 0; 
        QSharedPointer<QImage> m_lastImage = nullptr;
        std::unique_ptr<CombinedFrame> m_last_frame_info{nullptr};
        GPUImage flowVis;
        std::unique_ptr<VideoEncoder> videoEncoder{nullptr};


    protected:
        QString formatIndex(int index);
        void outputFrame(int i, QSharedPointer<QImage> q);
        bool loadFrame(int i);
        void retrieveOpticalFlow(int currentFrame);
        QSharedPointer<QImage> computeFlowVis(int currentFrame);
    
    signals:
        void frameReady(int64_t pts, QSharedPointer<QImage> image);

    public:
        StreamStabilizer(QDir originalVidPath,  QDir processedVidPath, std::optional<QString> stabilizedDirorVid,  
                                    std::optional<QString> modelType, int width, int height, int batchSize, bool streamingOutput);
        bool stabilizeAll();
        void startBackgroundThread();
        bool processOneFrame(int i);
        void quit();
        // return video_control
        VideoControl *getVideoControl() { return videoDecode.get_video_control(); }
        
        // get getters for m_lastPts and m_lastImage
        int64_t getLastPts() { return m_lastPts; }
        QSharedPointer<QImage> getLastImage() { return m_lastImage; }

};
