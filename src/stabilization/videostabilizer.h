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

#pragma once
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

#include "gpuimage.h"
#include "flowIO.h"
#include "imagehelpers.h"
#include "flowmodel.h"

struct hyperParams {
    float alpha;
    float beta;
    float gamma;
    int pyramidLevels;
    int numIter;
    float stepSize;
    float momFac;
} ;

class VideoStabilizer { 
  protected:
      int width;
      int height;
      int flowWidth;
      int flowHeight;
      int batchSize;
      // std::optional<pathInputs> inputPaths;
      // std::optional<QDir> opticalFlowDir;
      std::optional<QString> modelType;
      GPUImage stabilizedFrame;
      GPUImage lastStabilizedFrame;
      GPUImage flowFwd;
      GPUImage flowBwd;
      QList<QSharedPointer<QImage>> originalFramesQt;
      QList<QSharedPointer<GPUImage>> originalFrames;
      QList<QSharedPointer<GPUImage>> processedFrames;
      QList<QSharedPointer<GPUImage>> flowResultsFwd;
      QList<QSharedPointer<GPUImage>> flowResultsBwd;
      std::unique_ptr<OrtContext> ORT_CONTEXT;
      // std::unique_ptr<nmp::PythonContext> PYTHON_CONTEXT;
      std::vector<std::unique_ptr<IOInterface>> nmpInputImages;
      // std::unique_ptr<nmp::Model> pwcnetModel;
      std::unique_ptr<FlowModel> flowModel;
      QElapsedTimer timer;
      float timeOptFlow;
      float timeLoad;
      float timeStabilized;
      float timeSave;
      // stabilization
      GPUImage prevWarpIn;
      GPUImage prevWarpPr;
      GPUImage nextWarpIn;
      GPUImage nextWarpPr;
      GPUImage lastStabWarp;
      GPUImage consisOut;
      GPUImage consWt;
      GPUImage adapCmbIn;
      GPUImage adapCmbPr;
      QList<QSharedPointer<GPUImage>> pyrAdapCmbPr;
      QList<QSharedPointer<GPUImage>> pyrConsWt;
      QList<QSharedPointer<GPUImage>> pyrPr;
      QList<QSharedPointer<GPUImage>> pyrConsisOut;
      bool computeFlow;


      hyperParams controlParameters;
      void initHyperParams();

      QString formatIndex(int index);

      virtual bool loadFrame(int i) = 0; // to be implemented as stream loading or file loading
      virtual void outputFrame(int i, QSharedPointer<QImage> q) = 0; // to be implemented as stream output or file output
      void preloadProcessedFrames();
      void outputFinalFrames(int currentFrame);
      void retrieveOpticalFlow(int currentFrame);

    public:
        int k = 1;
        VideoStabilizer(int width, int height, int batchSize, std::optional<QString> modelType, bool computeFlow);
        bool doOneStep(int currentFrame);
        hyperParams *getHyperParams() { return &controlParameters;}
        // QSharedPointer<QImage> doOneStep(int currentFrame);
};