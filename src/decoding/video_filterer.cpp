// code adapted from https://github.com/pixop/video-compare
// Copyright (C) 2022 Jon Frydensbjerg (email: jon@pixop.com) 
// licensed under GNU General Public License v2.0
#include "video_filterer.h"
#include <iostream>
#include <string>
#include "ffmpeg.h"
#include "string_utils.h"

VideoFilterer::VideoFilterer(const Demuxer* demuxer, const VideoDecoder* video_decoder) : filter_graph_(avfilter_graph_alloc()) {
  std::vector<std::string> filters;

  // see FFmpeg documentation for more info on video filters
  //
  // filters.push_back("format=gray");
  // filters.push_back("yadif");

  if (demuxer->rotation() == 90) {
    filters.push_back("transpose=clock");
  } else if (demuxer->rotation() == 270) {
    filters.push_back("transpose=cclock");
  } else if (demuxer->rotation() == 180) {
    filters.push_back("hflip");
    filters.push_back("vflip");
  } else {
    filters.push_back("copy");
  }

  ffmpeg::check(init_filters(video_decoder->codec_context(), demuxer->time_base(), string_join(filters, ",")));
}

VideoFilterer::~VideoFilterer() {
  avfilter_graph_free(&filter_graph_);
}

int VideoFilterer::init_filters(const AVCodecContext* dec_ctx, const AVRational time_base, const std::string& filter_description) {
  AVFilterInOut* outputs = avfilter_inout_alloc();
  AVFilterInOut* inputs = avfilter_inout_alloc();

  int ret = 0;

  if ((outputs == nullptr) || (inputs == nullptr) || (filter_graph_ == nullptr)) {
    ret = AVERROR(ENOMEM);
  } else {
        // av_register_all();
    #if (LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 6, 102))
      avfilter_register_all();
    #endif

    // buffer video source: the decoded frames go here
    const AVFilter* buffersrc = avfilter_get_by_name("buffer");
    const std::string args =
        string_sprintf("video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d", dec_ctx->width, dec_ctx->height, dec_ctx->pix_fmt, time_base.num, time_base.den, dec_ctx->sample_aspect_ratio.num, dec_ctx->sample_aspect_ratio.den);

    std::cout << "VideoFilterer::init_filters: " << args << std::endl;
    ret = avfilter_graph_create_filter(&buffersrc_ctx_, buffersrc, "in", args.c_str(), nullptr, filter_graph_);
    if (ret < 0) {
      throw ffmpeg::Error{std::string("Cannot create buffer source, ") + std::to_string(ret)};
    }

    // buffer video sink: terminate the filter chain
    const AVFilter* buffersink = avfilter_get_by_name("buffersink");

    ret = avfilter_graph_create_filter(&buffersink_ctx_, buffersink, "out", nullptr, nullptr, filter_graph_);
    if (ret < 0) {
      throw ffmpeg::Error{"Cannot create buffer sink"};
    }

    outputs->name = av_strdup("in");
    outputs->filter_ctx = buffersrc_ctx_;
    outputs->pad_idx = 0;
    outputs->next = nullptr;

    inputs->name = av_strdup("out");
    inputs->filter_ctx = buffersink_ctx_;
    inputs->pad_idx = 0;
    inputs->next = nullptr;

    if ((ret = avfilter_graph_parse_ptr(filter_graph_, filter_description.c_str(), &inputs, &outputs, nullptr)) >= 0) {
      ret = avfilter_graph_config(filter_graph_, nullptr);
    }
  }

  avfilter_inout_free(&inputs);
  avfilter_inout_free(&outputs);

  return ret;
}

bool VideoFilterer::send(AVFrame* decoded_frame) {
  return av_buffersrc_add_frame_flags(buffersrc_ctx_, decoded_frame, AV_BUFFERSRC_FLAG_KEEP_REF) >= 0;
}

bool VideoFilterer::receive(AVFrame* filtered_frame) {
  auto ret = av_buffersink_get_frame(buffersink_ctx_, filtered_frame);

  if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
    return false;
  }
  ffmpeg::check(ret);
  return true;
}

size_t VideoFilterer::src_width() const {
  return buffersrc_ctx_->outputs[0]->w;
}

size_t VideoFilterer::src_height() const {
  return buffersrc_ctx_->outputs[0]->w;
}

AVPixelFormat VideoFilterer::src_pixel_format() const {
  return static_cast<AVPixelFormat>(buffersrc_ctx_->outputs[0]->format);
}

size_t VideoFilterer::dest_width() const {
  return buffersink_ctx_->inputs[0]->w;
}

size_t VideoFilterer::dest_height() const {
  return buffersink_ctx_->inputs[0]->h;
}

AVPixelFormat VideoFilterer::dest_pixel_format() const {
  return static_cast<AVPixelFormat>(buffersink_ctx_->inputs[0]->format);
}
