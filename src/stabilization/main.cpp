/*  CLI for video stabilization

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

#include <QCoreApplication>
#include <QCommandLineParser>
#include <QDebug>
#include <csignal>
#include <iostream>
#include <optional>

// #include "stabilize.h"
#include "stabilizefiles.h"
#include "stabilizestream.h"
#include "../decoding/demuxer.h"
#include "../decoding/video_decoder.h"


void signalHandler( int signum ) {
   std::cout << "Interrupt signal (" << signum << ") received.\n";
   exit(signum);  
}

int main(int argc, char *argv[])
{
    // register signal SIGINT and signal handler  
    signal(SIGINT, signalHandler);  

    QCoreApplication app(argc, argv);

    QCommandLineParser parser;
    QCommandLineOption helpOption = parser.addHelpOption();
    parser.addPositionalArgument("originalFrames", "Original frame directory");
    parser.addPositionalArgument("processedFrames", "Processed frames directory");
    parser.addPositionalArgument("stabilizedFrames", "Output directory for stabilized frames");

    QCommandLineOption computeOption(QStringList() << "c" << "compute",
            QCoreApplication::translate("main", "Compute optical flow using <model>. One of ['pwcnet', 'pwcnet-light']"),
            QCoreApplication::translate("main", "model"));
    parser.addOption(computeOption);

    QCommandLineOption batchsizeOption(QStringList() << "b" << "batchsize",
            QCoreApplication::translate("main", "Batchsize for optical flow computation. Default is 1."),
            QCoreApplication::translate("main", "batchsize"),
            QCoreApplication::translate("main", "1"));
    parser.addOption(batchsizeOption);


    QCommandLineOption flowDirectoryOption(QStringList() << "f" << "flow-directory",
            QCoreApplication::translate("main", "Read flow from <directory>."),
            QCoreApplication::translate("main", "directory"));
    parser.addOption(flowDirectoryOption);

    parser.process(app);
    const QStringList& args = parser.positionalArguments();

    if (parser.isSet(helpOption) || args.size() != 3) {
        parser.showHelp();
        return 1;
    }

    if (parser.isSet(computeOption) == parser.isSet(flowDirectoryOption)) {
        std::cerr << "Error: Either compute option (-c) or flow directory option (-f) have to be set." << std::endl << std::endl;
        parser.showHelp();
        return 1;
    }

    auto modelChoices = QStringList() << "pwcnet" << "pwcnet-light";
    if (parser.isSet(computeOption) && !modelChoices.contains(parser.value(computeOption))) {
        std::cerr << "Unknown optical flow model '" << parser.value(computeOption).toStdString().c_str() << "' specified." << std::endl << std::endl;

        parser.showHelp();
        return 1;
    }

    bool toIntSuccess;
    int batchSize = parser.value(batchsizeOption).toInt(&toIntSuccess);
    if (!toIntSuccess || batchSize < 1) {
        std::cout << "Batchsize [-b] must be a positive int" << std::endl << std::endl;
        parser.showHelp();
        return 1;  
    }


    qDebug() << args;
    QString originalFrameDir = args.at(0);
    QString processedFrameDir = args.at(1);
    QString stabilizedFrameDir = args.at(2);

    
    std::optional<QString> opticalFlowDir = parser.isSet(flowDirectoryOption) ? std::make_optional(parser.value(flowDirectoryOption)) : std::nullopt;
    std::optional<QString> computeModel = parser.isSet(computeOption) ?  std::make_optional(parser.value(computeOption)) : std::nullopt;

    // check that originalFrameDir and processedFrameDir both exist and are directories or both video files
    if (!QFileInfo(originalFrameDir).exists()) {
        std::cerr << "Original frame directory / video does not exist." << std::endl;
        return 1;
    }
    if (!QFileInfo(processedFrameDir).exists()) {
        std::cerr << "processedFrameDir directory / video does not exist." << std::endl;
        return 1;
    }

    bool originalIsDir = QFileInfo(originalFrameDir).isDir();
    bool processedIsDir = QFileInfo(processedFrameDir).isDir();
    if (originalIsDir != processedIsDir) {
        std::cerr << "originalFrameDir and processedFrameDir have to be both directories or both video files." << std::endl;
        return 1;
    }


    if (originalIsDir) {
        auto inpname = QDir(originalFrameDir).entryList(QStringList({"*.png", "*.jpg"}), QDir::Files, QDir::Name)[0];
        auto inputfirst = QDir(originalFrameDir).filePath(inpname); 
        auto out = inputfirst.toStdString();
        std::cout << "Loading first frame " << inputfirst.toStdString() << std::endl;
        QImage image;
        bool loaded = image.load(inputfirst);
        int width = image.width();
        int height = image.height();
        Q_ASSERT(loaded);
        assert(width != 0 && height != 0);

        FileStabilizer fs(originalFrameDir, processedFrameDir, stabilizedFrameDir, opticalFlowDir, computeModel, width, height, batchSize, parser.isSet(computeOption));
        return fs.stabilizeAll();
    } else {
        auto inputVideo = QDir(originalFrameDir);
        // // check if input video exists and is file
        // if (!inputVideo.exists()) {
        //     std::cerr << "Input video does not exist." << std::endl;
        //     return 1;
        // }
        if (!QFileInfo(originalFrameDir).isFile()) {
            std::cerr << "Input video has to be a file." << std::endl;
            return 1;
        }


        auto demuxer = new Demuxer(originalFrameDir.toStdString());
        auto decoder = new VideoDecoder(demuxer->video_codec_parameters());
        int width = decoder->width();
        int height = decoder->height();
        delete decoder;
        delete demuxer;


        // check if stabilizedFrameDir is a filename with a common video extension
        if (!QFileInfo(stabilizedFrameDir).isDir()) {
            auto ext = QFileInfo(stabilizedFrameDir).suffix();
            // check if stabilzedDir contains a dot, if not it is a directory, and we don't need to check the extension. Create the directory
            if (ext == "") {
                QDir().mkdir(stabilizedFrameDir);
            } else if (ext != "mp4" && ext != "avi" && ext != "mkv" && ext != "mov" && ext != "webm" && ext != "wmv" && ext != "flv" && ext != "m4v") {
                std::cerr << "stabilizedFrameDir has to be a directory or a video file with a common video extension." << std::endl;
                std::cerr << "Allowed extensions are: mp4, avi, mkv, mov, webm, wmv, flv, m4v" << std::endl;
                return 1;
            }
        }

        StreamStabilizer ss(originalFrameDir, processedFrameDir, stabilizedFrameDir, computeModel, width, height, batchSize, false);
        return ss.stabilizeAll();
    }

}
