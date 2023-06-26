// code adapted from https://github.com/pixop/video-compare
// Copyright (C) 2022 Jon Frydensbjerg (email: jon@pixop.com) 
// licensed under GNU General Public License v2.0
#pragma once
#include <QObject>
#include <functional>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <queue>
#include "queue.h"

enum class VideoType {
    ORIGINAL,
    STABILIZED,
    STYLIZED, 
    FLOWVIS
};


class VideoControl : public QObject { 
    Q_OBJECT

  private:
    /* data */
    int64_t seek_absolute_ = 0.0;
    int64_t frame_offset_delta_ = 0;
    bool seek_from_start_ = false;
    bool quit_ = false;
    bool play_ = true;
    bool is_seeking_ = false;
    int64_t duration_ = 0;
    int64_t current_position_ = 0;
    VideoType video_type_ = VideoType::STABILIZED;

  signals:
    void quit_signal();
  
  public:
    bool get_quit() { return quit_; }
    bool get_play() { return play_; } 
    int64_t get_seek_absolute() { return seek_absolute_;}
    bool get_seek_from_start() { return seek_from_start_; }
    int get_frame_offset_delta() { return frame_offset_delta_; }
    int64_t get_duration() { return duration_; }
    bool get_is_seeking() { return is_seeking_; } 
    VideoType get_video_type() { return video_type_; }

    int get_duration_in_milli_seconds() {
        return duration_ / 1000;
    }



    int64_t get_current_position() { return current_position_; }

    void set_seek_absolute(int64_t seek_absolute) { seek_absolute_ = seek_absolute; }
    void set_seek_from_start(bool seek_from_start) { seek_from_start_ = seek_from_start; }
    void set_frame_offset_delta(int frame_offset_delta) { frame_offset_delta_ = frame_offset_delta; }
    void set_quit(bool quit) { quit_ = quit; emit quit_signal(); }
    void set_play(bool play) { play_ = play; }
    void set_duration(int64_t duration) { duration_ = duration; }
    void set_current_position(int64_t current_position) { current_position_ = current_position; }
    void set_seeking(bool is_seeking) { is_seeking_ = is_seeking; }
    void set_video_type(VideoType video_type) { video_type_ = video_type;}
  };

  
  
