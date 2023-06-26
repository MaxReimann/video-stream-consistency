// code adapted from https://github.com/pixop/video-compare
// Copyright (C) 2022 Jon Frydensbjerg (email: jon@pixop.com) 
// licensed under GNU General Public License v2.0
#include "video_decoder.h"
#include <iostream>
#include <string>
#include "ffmpeg.h"

VideoDecoder::VideoDecoder(AVCodecParameters* codec_parameters) {
#if (LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 6, 102))
  avcodec_register_all();
#endif
  const auto* const codec = avcodec_find_decoder(codec_parameters->codec_id);
  if (codec == nullptr) {
    throw ffmpeg::Error{"Unsupported video codec"};
  }
  codec_context_ = avcodec_alloc_context3(codec);
  if (codec_context_ == nullptr) {
    throw ffmpeg::Error{"Couldn't allocate video codec context"};
  }
  ffmpeg::check(avcodec_parameters_to_context(codec_context_, codec_parameters));
  ffmpeg::check(avcodec_open2(codec_context_, codec, nullptr));
}

VideoDecoder::~VideoDecoder() {
  avcodec_free_context(&codec_context_);
}

bool VideoDecoder::send(AVPacket* packet) {
  auto ret = avcodec_send_packet(codec_context_, packet);
  if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
    return false;
  }
  ffmpeg::check(ret);
  return true;
}

bool VideoDecoder::receive(AVFrame* frame) {
  auto ret = avcodec_receive_frame(codec_context_, frame);
  if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
    return false;
  }
  ffmpeg::check(ret);
  return true;
}

void VideoDecoder::flush() {
  avcodec_flush_buffers(codec_context_);
}

unsigned VideoDecoder::width() const {
  return codec_context_->width;
}

unsigned VideoDecoder::height() const {
  return codec_context_->height;
}

AVPixelFormat VideoDecoder::pixel_format() const {
  return codec_context_->pix_fmt;
}

AVRational VideoDecoder::time_base() const {
  return codec_context_->time_base;
}

AVCodecContext* VideoDecoder::codec_context() const {
  return codec_context_;
}