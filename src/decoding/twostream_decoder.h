// code adapted from https://github.com/pixop/video-compare
// Original Copyright (C) 2022 Jon Frydensbjerg 
// Modified by 2023 Max Reimann (max.reimann@hpi.de)
// licensed under GNU General Public License v2.0
#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <QObject>
#include <QImage>
#include <QSharedPointer>
#include "demuxer.h"
// #include "display.h"
#include "format_converter.h"
#include "queue.h"
#include "timer.h"
#include "video_decoder.h"
#include "video_filterer.h"
#include "video_control.h"
extern "C" {
#include <libavcodec/avcodec.h>
}

struct AVFrame;



struct CombinedFrame
{
  int frame_offset;
  int64_t pts;
  QSharedPointer<QImage> original;
  QSharedPointer<QImage> processed;
};


class TwoStreamDecoder : public QObject { 
    Q_OBJECT

 public:
  TwoStreamDecoder(double time_shift_ms, const std::string& left_file_name, const std::string& right_file_name);
  void run();

  VideoControl* get_video_control() { return videoControl.get(); }
  std::exception_ptr get_exception() { return exception_; }
  CombinedFrameQueue& get_frame_queue() { return combined_frame_queue; }

public slots:
    void quit();

 private:
  void thread_demultiplex_left();
  void thread_demultiplex_right();
  void demultiplex(int video_idx);

  void thread_decode_video_left();
  void thread_decode_video_right();
  bool process_packet(int video_idx, AVPacket* packet, AVFrame* frame_decoded);
  void decode_video(int video_idx);
  void video();

  double time_shift_ms_;
  std::unique_ptr<Demuxer> demuxer_[2];
  std::unique_ptr<VideoDecoder> video_decoder_[2];
  std::unique_ptr<VideoFilterer> video_filterer_[2];

  size_t max_width_;
  size_t max_height_;
  double shortest_duration_;
  std::unique_ptr<FormatConverter> format_converter_[2];
  std::unique_ptr<VideoControl> videoControl;
  // std::unique_ptr<Display> display_;
  std::unique_ptr<Timer> timer_;
  std::unique_ptr<PacketQueue> packet_queue_[2];
  std::unique_ptr<FrameQueue> frame_queue_[2];
  std::vector<std::thread> stages_;
  static const size_t QUEUE_SIZE;
  std::exception_ptr exception_{};
  volatile bool seeking_{false};
  volatile bool readyToSeek_[2][2];
  CombinedFrameQueue combined_frame_queue;
  
};
