// code adapted from https://github.com/pixop/video-compare
// Copyright (C) 2022 Jon Frydensbjerg (email: jon@pixop.com) 
// licensed under GNU General Public License v2.0

#include "demuxer.h"
#include <iostream>
#include "ffmpeg.h"

Demuxer::Demuxer(const std::string& file_name) {
#if (LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 6, 102))
  av_register_all();
#endif
  ffmpeg::check(file_name, avformat_open_input(&format_context_, file_name.c_str(), nullptr, nullptr));

  format_context_->probesize = 100000000;
  format_context_->max_analyze_duration = 100000000;

  ffmpeg::check(file_name, avformat_find_stream_info(format_context_, nullptr));
  video_stream_index_ = ffmpeg::check(file_name, av_find_best_stream(format_context_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0));
}

Demuxer::~Demuxer() {
  avformat_close_input(&format_context_);
}

AVCodecParameters* Demuxer::video_codec_parameters() {
  return format_context_->streams[video_stream_index_]->codecpar;
}

int Demuxer::video_stream_index() const {
  return video_stream_index_;
}

AVRational Demuxer::time_base() const {
  return format_context_->streams[video_stream_index_]->time_base;
}

int64_t Demuxer::duration() const {
  return format_context_->duration;
}

int Demuxer::rotation() const {
  double theta = 0;

  uint8_t* displaymatrix = av_stream_get_side_data(format_context_->streams[video_stream_index_], AV_PKT_DATA_DISPLAYMATRIX, nullptr);

  if (displaymatrix != nullptr) {
    theta = -av_display_rotation_get(reinterpret_cast<int32_t*>(displaymatrix));
  }

  theta -= 360 * floor(theta / 360 + 0.9 / 360);

  return theta;
}

bool Demuxer::operator()(AVPacket& packet) {
  return av_read_frame(format_context_, &packet) >= 0;
}

bool Demuxer::seek(const float position, const bool backward) {
  int64_t seek_target = static_cast<int64_t>(position * AV_TIME_BASE);

    if (avformat_seek_file(format_context_, -1, 0, seek_target, INT64_MAX, 0) < 0) { 
        av_log(NULL, AV_LOG_ERROR, "ERROR av_seek_frame: %u\n", static_cast<unsigned int>(seek_target));
        return false;
    } else {
        // av_log(NULL, AV_LOG_ERROR, "SUCCEEDED av_seek_frame: %u newPos:%d\n", static_cast<unsigned int>(seek_target), static_cast<int>(format_context_->pb->pos));
        // avcodec_flush_buffers(format_context_->streams[video_stream_index_]->codec);
        return true;
    }
}

