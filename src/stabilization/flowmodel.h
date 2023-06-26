#pragma once
#include <QImage>
#include <QSharedPointer>
#include <QElapsedTimer>
#include "gpuimage.h"

#include <inference/OrtContext.h>
#include <inference/InferenceModelVariant.h>
// #include <nmp/PythonContext.h>
// #include <nmp/Model.h>
// #include <nmp/model/ImageArrayIOHelper.h>

struct flowTiming
{
    qint64 loadTime;
    qint64 runTime;
};

class FlowModel
{
private:
    std::unique_ptr<InferenceModelVariant> model;
    std::unique_ptr<InferenceModelVariant> flowvismodel;
    // std::vector<std::unique_ptr<IOInterface>> nmpInputImages;
    std::vector<std::shared_ptr<IOInterface>> nmpInputImages;
    std::vector<std::shared_ptr<IOInterface>> adjustedWidthHeight;
    std::shared_ptr<IOInterface> outputFlow;
    std::shared_ptr<IOInterface> flowVis;
    QElapsedTimer timer;
    OrtContext* _ort_context;
    // PythonContext* _python_context;
    int batchSize;
    int width;
    int height;
public:
    FlowModel(QString modelChoice, OrtContext* ort_context,  int batchSize, int width, int height);
    void run(QList<QSharedPointer<QImage>>& originalFramesQt,  QList<QSharedPointer<GPUImage>>& results, 
        int indexFirst, int indexSecond, flowTiming* timing);
    void runFlowVis(QList<QSharedPointer<GPUImage>> &results, flowTiming *timing);
};
        