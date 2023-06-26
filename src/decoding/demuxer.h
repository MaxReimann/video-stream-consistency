// code adapted from https://github.com/pixop/video-compare
// Copyright (C) 2022 Jon Frydensbjerg (email: jon@pixop.com) 
// licensed under GNU General Public License v2.0
#pragma once
#include <string>
extern "C" {
#include "libavformat/avformat.h"
#include "libavutil/display.h"
}

class Demuxer {
 public:
  explicit Demuxer(const std::string& file_name);
  ~Demuxer();
  AVCodecParameters* video_codec_parameters();
  int video_stream_index() const;
  AVRational time_base() const;
  int64_t duration() const;
  int rotation() const;
  bool operator()(AVPacket& packet);
  bool seek(float position, bool backward);

 private:
  AVFormatContext* format_context_{};
  int video_stream_index_{};
};
