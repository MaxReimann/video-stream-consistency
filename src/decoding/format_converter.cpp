// code adapted from https://github.com/pixop/video-compare
// Copyright (C) 2022 Jon Frydensbjerg (email: jon@pixop.com) 
// licensed under GNU General Public License v2.0
#include "format_converter.h"
#include <iostream>

extern "C" {
  #include <libavutil/imgutils.h>
}

FormatConverter::FormatConverter(size_t src_width, size_t src_height, size_t dest_width, size_t dest_height, AVPixelFormat input_pixel_format, AVPixelFormat output_pixel_format)
    : src_width_{src_width},
      src_height_{src_height},
      dest_width_{dest_width},
      dest_height_{dest_height},
      output_pixel_format_{output_pixel_format}
      {
        // AVCodecContext* pCodecCtx = _videoStream->codec;
        // from https://stackoverflow.com/questions/23067722/swscaler-warning-deprecated-pixel-format-used
        AVPixelFormat pixFormat;
        bool changeColorspaceDetails = false;
        switch (input_pixel_format)
          {
            case AV_PIX_FMT_YUVJ420P:
              pixFormat = AV_PIX_FMT_YUV420P;
              changeColorspaceDetails = true;
              break;
            case AV_PIX_FMT_YUVJ422P:
              pixFormat = AV_PIX_FMT_YUV422P;
              changeColorspaceDetails = true;
              break;
            case AV_PIX_FMT_YUVJ444P:
              pixFormat = AV_PIX_FMT_YUV444P;
              changeColorspaceDetails = true;
              break;
            case AV_PIX_FMT_YUVJ440P:
              pixFormat = AV_PIX_FMT_YUV440P;
              changeColorspaceDetails = true;
              break;
            default:
              pixFormat = input_pixel_format;
          }
          // initialize SWS context for software scaling
          conversion_context_ = sws_getContext(
              // Source
              src_width,
              src_height,
              pixFormat,
              // Destination
              dest_width,
              dest_height,
              output_pixel_format,
              // Filters
              SWS_BILINEAR,    // SWS_BICUBIC,
              nullptr,
              nullptr,
              nullptr);

          if (changeColorspaceDetails)
          {
              // change the range of input data by first reading the current color space and then setting it's range as yuvj.
              int dummy[4];
              int srcRange, dstRange;
              int brightness, contrast, saturation;
              sws_getColorspaceDetails(conversion_context_, (int**)&dummy, &srcRange, (int**)&dummy, &dstRange, &brightness, &contrast, &saturation);
              const int* coefs = sws_getCoefficients(SWS_CS_DEFAULT);
              srcRange = 1; // this marks that values are according to yuvj
              sws_setColorspaceDetails(conversion_context_, coefs, srcRange, coefs, dstRange,
                                        brightness, contrast, saturation);
          }

  }

size_t FormatConverter::src_width() const {
  return src_width_;
}

size_t FormatConverter::src_height() const {
  return src_height_;
}

size_t FormatConverter::dest_width() const {
  return dest_width_;
}

size_t FormatConverter::dest_height() const {
  return dest_height_;
}

AVPixelFormat FormatConverter::output_pixel_format() const {
  return output_pixel_format_;
}

// convert the image from its native format to QImage
void FormatConverter::operator()(AVFrame* src, QImage* dst) {
  int linesizes[4];
  av_image_fill_linesizes(linesizes, output_pixel_format_, dest_width_);
  auto p = dst->bits();
  uint8_t *dst_byte[] = {p};

  sws_scale(conversion_context_, 
      // Source
      src->data, src->linesize, 0, src_height_,
      // Destination
       dst_byte, linesizes);
}
