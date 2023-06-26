// code adapted from https://github.com/pixop/video-compare
// Copyright (C) 2022 Jon Frydensbjerg (email: jon@pixop.com) 
// licensed under GNU General Public License v2.0
#pragma once
#include <QImage>
#include <QSharedPointer>

extern "C" {
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
}

struct TimedQFrame
{
  int64_t pts;
  QSharedPointer<QImage> frame;
};

class FormatConverter {
 public:
  FormatConverter(size_t src_width, size_t src_height, size_t dest_width, size_t dest_height, AVPixelFormat input_pixel_format, AVPixelFormat output_pixel_format);
  size_t src_width() const;
  size_t src_height() const;
  size_t dest_width() const;
  size_t dest_height() const;
  AVPixelFormat output_pixel_format() const;
  // void operator()(AVFrame* src, AVFrame* dst);
  void operator()(AVFrame* src, QImage* dst);

 private:
  size_t src_width_;
  size_t src_height_;
  size_t dest_width_;
  size_t dest_height_;
  AVPixelFormat output_pixel_format_;
  SwsContext* conversion_context_{};
};
