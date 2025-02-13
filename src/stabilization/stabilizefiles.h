
#include "videostabilizer.h"

#include <QDir>
#include <optional>
#include <QStringList>
#include <QDebug>


struct pathInputs
{
    QStringList originalFramePaths;
    QStringList processedFramePaths;
    int frameCount;
};


class FileStabilizer : protected VideoStabilizer
{
    private:
        QDir originalFrameDir;
        QDir processedFrameDir;
        QDir stabilizedFrameDir;
        std::optional<QDir> opticalFlowDir;
        std::optional<pathInputs> inputPaths;
        int frameCount;
    protected:
        QString formatIndex(int index);
        bool loadFrame(int i);
        void outputFrame(int i, QSharedPointer<QImage> q);
        void retrieveOpticalFlow(int currentFrame);
    public:
        // FileStabilizer(QDir originalFrameDir, QDir processedFrameDir, QDir stabilizedFrameDir, std::optional<QDir> opticalFlowDir, 
        //                 std::optional<QString> modelType, int width, int height, int batchSize, bool computeOpticalFlow,
        //                 const QString &configFilePath);
        FileStabilizer(QDir originalFrameDir, QDir processedFrameDir, QDir stabilizedFrameDir, std::optional<QDir> opticalFlowDir, 
                        std::optional<QString> modelType, int width, int height, int batchSize, bool computeOpticalFlow);
        bool stabilizeAll();
};
