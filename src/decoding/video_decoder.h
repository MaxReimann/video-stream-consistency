// code adapted from https://github.com/pixop/video-compare
// Copyright (C) 2022 Jon Frydensbjerg (email: jon@pixop.com) 
// licensed under GNU General Public License v2.0
#pragma once
extern "C" {
#include "libavcodec/avcodec.h"
}

class VideoDecoder {
 public:
  explicit VideoDecoder(AVCodecParameters* codec_parameters);
  ~VideoDecoder();
  bool send(AVPacket* packet);
  bool receive(AVFrame* frame);
  void flush();
  bool swap_dimensions() const;
  unsigned width() const;
  unsigned height() const;
  AVPixelFormat pixel_format() const;
  AVRational time_base() const;
  AVCodecContext* codec_context() const;

 private:
  AVCodecContext* codec_context_{};
};
