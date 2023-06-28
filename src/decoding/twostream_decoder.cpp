// code adapted from https://github.com/pixop/video-compare
// Original Copyright (C) 2022 Jon Frydensbjerg 
// Modified by 2023 Max Reimann (max.reimann@hpi.de)
// licensed under GNU General Public License v2.0
#include "twostream_decoder.h"
#include <algorithm>
#include <chrono>
#include <deque>
#include <iostream>
#include <thread>
#include "ffmpeg.h"
#include "sorted_flat_deque.h"
#include "string_utils.h"
extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/time.h>
}

const size_t TwoStreamDecoder::QUEUE_SIZE{5};

static inline bool is_behind(int64_t frame1_pts, int64_t frame2_pts, int64_t delta_pts) {
  float t1 = static_cast<float>(frame1_pts) * AV_TIME_TO_SEC;
  float t2 = static_cast<float>(frame2_pts) * AV_TIME_TO_SEC;
  float delta_s = static_cast<float>(delta_pts) * AV_TIME_TO_SEC;

  float diff = t1 - t2;
  float tolerance = std::max(delta_s, 1.0F / 120.0F);

  return diff < -tolerance;
}



TwoStreamDecoder::TwoStreamDecoder(const double time_shift_ms, const std::string& left_file_name, const std::string& right_file_name)
    : time_shift_ms_(time_shift_ms),
      demuxer_{std::make_unique<Demuxer>(left_file_name), std::make_unique<Demuxer>(right_file_name)},
      video_decoder_{std::make_unique<VideoDecoder>(demuxer_[0]->video_codec_parameters()), std::make_unique<VideoDecoder>(demuxer_[1]->video_codec_parameters())},
      video_filterer_{std::make_unique<VideoFilterer>(demuxer_[0].get(), video_decoder_[0].get()), std::make_unique<VideoFilterer>(demuxer_[1].get(), video_decoder_[1].get())},
      max_width_{std::max(video_filterer_[0]->dest_width(), video_filterer_[1]->dest_width())},
      max_height_{std::max(video_filterer_[0]->dest_height(), video_filterer_[1]->dest_height())},
      shortest_duration_{std::min(demuxer_[0]->duration(), demuxer_[1]->duration()) * AV_TIME_TO_SEC},
      format_converter_{std::make_unique<FormatConverter>(video_filterer_[0]->dest_width(), video_filterer_[0]->dest_height(), max_width_, max_height_, video_filterer_[0]->dest_pixel_format(), AV_PIX_FMT_RGBA),
                        std::make_unique<FormatConverter>(video_filterer_[1]->dest_width(), video_filterer_[1]->dest_height(), max_width_, max_height_, video_filterer_[1]->dest_pixel_format(), AV_PIX_FMT_RGBA)},
      timer_{std::make_unique<Timer>()},
      packet_queue_{std::make_unique<PacketQueue>(QUEUE_SIZE), std::make_unique<PacketQueue>(QUEUE_SIZE)},
      frame_queue_{std::make_unique<FrameQueue>(QUEUE_SIZE), std::make_unique<FrameQueue>(QUEUE_SIZE)},
      combined_frame_queue(CombinedFrameQueue(QUEUE_SIZE)),
      videoControl(std::make_unique<VideoControl>()) 
  {
      videoControl->set_duration(demuxer_[0]->duration());
      connect(videoControl.get(), &VideoControl::quit_signal, this, &TwoStreamDecoder::quit);
  }

void TwoStreamDecoder::run() {
  stages_.emplace_back(&TwoStreamDecoder::thread_demultiplex_left, this);
  stages_.emplace_back(&TwoStreamDecoder::thread_demultiplex_right, this);
  stages_.emplace_back(&TwoStreamDecoder::thread_decode_video_left, this);
  stages_.emplace_back(&TwoStreamDecoder::thread_decode_video_right, this);
  // stages_.emplace_back(&VideoCompare::video, this);
  video();

  for (auto& stage : stages_) {
    stage.join();
  }
}

void TwoStreamDecoder::thread_demultiplex_left() {
  demultiplex(0);
}

void TwoStreamDecoder::thread_demultiplex_right() {
  demultiplex(1);
}

void TwoStreamDecoder::demultiplex(const int video_idx) {
  try {
    for (;;) {
      if (seeking_ && readyToSeek_[1][video_idx]) {
        readyToSeek_[0][video_idx] = true;

        std::chrono::milliseconds sleep(10);
        std::this_thread::sleep_for(sleep);
        continue;
      }

      // Create AVPacket
      std::unique_ptr<AVPacket, std::function<void(AVPacket*)>> packet{new AVPacket, [](AVPacket* p) {
                                                                         av_packet_unref(p);
                                                                         delete p;
                                                                       }};
      av_init_packet(packet.get());
      packet->data = nullptr;

      
      if (videoControl->get_quit()) {
        break;
      }

      // Read frame into AVPacket
      if (!(*demuxer_[video_idx])(*packet)) {
        packet_queue_[video_idx]->finished();
        break;
      }

      // Move into queue if first video stream
      if (packet->stream_index == demuxer_[video_idx]->video_stream_index()) {
        if (!packet_queue_[video_idx]->push(std::move(packet))) {
          break;
        }
      }
    }
  } catch (...) {
    exception_ = std::current_exception();
    frame_queue_[video_idx]->quit();
    packet_queue_[video_idx]->quit();
    combined_frame_queue.quit();
  }
}

void TwoStreamDecoder::thread_decode_video_left() {
  decode_video(0);
}

void TwoStreamDecoder::thread_decode_video_right() {
  decode_video(1);
}

bool TwoStreamDecoder::process_packet(const int video_idx, AVPacket* packet, AVFrame* frame_decoded) {
  const AVRational microseconds = {1, AV_TIME_BASE};

  bool sent = video_decoder_[video_idx]->send(packet);

  // If a whole frame has been decoded,
  // adjust time stamps and add to queue
  while (video_decoder_[video_idx]->receive(frame_decoded)) {
    frame_decoded->pts = av_rescale_q(frame_decoded->pts, demuxer_[video_idx]->time_base(), microseconds);

    // send decodes frame to filterer
    if (!video_filterer_[video_idx]->send(frame_decoded)) {
      throw std::runtime_error("Error while feeding the filtergraph");
    }

    std::unique_ptr<AVFrame, std::function<void(AVFrame*)>> frame_filtered{av_frame_alloc(), [](AVFrame* f) { av_free(f->data[0]); }};

    while (true) {
      // get next filtered frame
      if (!video_filterer_[video_idx]->receive(frame_filtered.get())) {
        break;
      }
      
      std::unique_ptr<TimedQFrame> frame_converted = std::make_unique<TimedQFrame>();
      frame_converted->pts = frame_filtered->pts;
      frame_converted->frame = QSharedPointer<QImage>(new QImage(max_width_, max_height_, QImage::Format_RGBA8888));
      (*format_converter_[video_idx])(frame_filtered.get(), frame_converted->frame.get());

      av_frame_unref(frame_filtered.get());
      if (!frame_queue_[video_idx]->push(std::move(frame_converted))) {
        return sent;
      }
    }
  }

  return sent;
}

void TwoStreamDecoder::decode_video(const int video_idx) {
  try {
    for (;;) {
      // Create AVFrame and AVPacket
      std::unique_ptr<AVFrame, std::function<void(AVFrame*)>> frame_decoded{av_frame_alloc(), [](AVFrame* f) { av_frame_free(&f); }};
      std::unique_ptr<AVPacket, std::function<void(AVPacket*)>> packet{nullptr, [](AVPacket* p) {
                                                                         av_packet_unref(p);
                                                                         delete p;
                                                                       }};

      if (videoControl->get_quit()) {
        break;
      }

      // Read packet from queue
      if (!packet_queue_[video_idx]->pop(packet)) {
        // Flush remaining frames cached in the decoder
        while (process_packet(video_idx, packet.get(), frame_decoded.get())) {
          ;
        }

        frame_queue_[video_idx]->finished();
        break;
      }

      if (seeking_) {
        video_decoder_[video_idx]->flush();

        readyToSeek_[1][video_idx] = true;

        std::chrono::milliseconds sleep(10);
        std::this_thread::sleep_for(sleep);

        const AVRational microseconds = {1, AV_TIME_BASE};
        bool seekDecoding = true;
        // decode frames until reaching videoControl->get_seek_absolute() time
        while (packet_queue_[video_idx]->pop(packet)) {
          bool sent = video_decoder_[video_idx]->send(packet.get());
          while (video_decoder_[video_idx]->receive(frame_decoded.get())) {
            frame_decoded->pts = av_rescale_q(frame_decoded->pts, demuxer_[video_idx]->time_base(), microseconds);
            
            if (frame_decoded->pts >= videoControl->get_seek_absolute() - MILLISEC_TO_AV_TIME * 5) {
              seekDecoding = false;
              break;
            }
          }
          if (!seekDecoding) {
            break;
          }
        }
        continue;
      }

      // If the packet didn't send, receive more frames and try again
      while (!process_packet(video_idx, packet.get(), frame_decoded.get()) && !seeking_) {
        ;
      }
    }
  } catch (...) {
    exception_ = std::current_exception();
    frame_queue_[video_idx]->quit();
    packet_queue_[video_idx]->quit();
    combined_frame_queue.quit();
  }
}

void TwoStreamDecoder::video() {
  try {
    std::deque<std::unique_ptr<TimedQFrame>> left_frames;
    std::deque<std::unique_ptr<TimedQFrame>> right_frames;
    int frame_offset = 0;

    std::unique_ptr<TimedQFrame> frame_left{nullptr};
    std::unique_ptr<TimedQFrame> frame_right{nullptr};

    int64_t left_pts = 0;
    int64_t left_decoded_picture_number = 0;
    int64_t left_previous_decoded_picture_number = -1;
    int64_t delta_left_pts = 0;
    int64_t right_pts = 0;
    int64_t right_decoded_picture_number = 0;
    int64_t right_previous_decoded_picture_number = -1;
    int64_t delta_right_pts = 0;

    sorted_flat_deque<int32_t> left_deque(8);
    sorted_flat_deque<int32_t> right_deque(8);

    int64_t right_time_shift = time_shift_ms_ * MILLISEC_TO_AV_TIME;

    for (uint64_t frame_number = 0;; ++frame_number) {
      std::string error_message;

      float current_position = left_pts * AV_TIME_TO_SEC;

      if (videoControl->get_is_seeking()) { //} && (videoControl->get_seek_absolute() / 1e6 != videoControl->get_current_position() / 1e6)) {
        std::cout << "start of 2strdecod loop. Seeking to " << videoControl->get_seek_absolute() / 1000 << "ms" << std::endl;

        if (packet_queue_[0]->is_finished() || packet_queue_[1]->is_finished()) {
          error_message = "Unable to perform seek (end of file reached)";
        } else {
          // compute effective time shift
          right_time_shift = time_shift_ms_ * MILLISEC_TO_AV_TIME; // + total_right_time_shifted * (delta_right_pts > 0 ? delta_right_pts : 10000);

          // round away from zero to nearest 2 ms
          if (right_time_shift > 0) {
            right_time_shift = ((right_time_shift / 1000) + 2) * 1000;
          } else if (right_time_shift < 0) {
            right_time_shift = ((right_time_shift / 1000) - 2) * 1000;
          }

          seeking_ = true;
          readyToSeek_[0][0] = false;
          readyToSeek_[0][1] = false;
          readyToSeek_[1][0] = false;
          readyToSeek_[1][1] = false;

          while (true) {
            bool all_empty = true;

            all_empty = all_empty && readyToSeek_[0][0];
            all_empty = all_empty && readyToSeek_[0][1];
            all_empty = all_empty && readyToSeek_[1][0];
            all_empty = all_empty && readyToSeek_[1][1];

            if (all_empty) {
              break;
            }
            frame_queue_[0]->empty();
            frame_queue_[1]->empty();
          }

          packet_queue_[0]->empty();
          packet_queue_[1]->empty();
          frame_queue_[0]->empty();
          frame_queue_[1]->empty();
          combined_frame_queue.empty();

          float next_position;

          if (videoControl->get_seek_from_start()) {
            // seek from start based on the shortest stream duration in seconds
            next_position = shortest_duration_ * videoControl->get_seek_absolute() / AV_TIME_BASE;
          } else {
            // next_position = current_position + videoControl->get_seek_absolute();
            next_position = static_cast<double>(videoControl->get_seek_absolute()) * AV_TIME_TO_SEC;
          }

          // bool backward = ((next_position - current_position) < 0.0F);
          std::cout << "next_position Seeking to " << next_position << "s" << std::endl;
          std::cout << "get_current_position (displayed)" << videoControl->get_current_position() << " pts " << videoControl->get_current_position() * AV_TIME_TO_SEC << "s" << std::endl;

          if (!demuxer_[0]->seek(std::max(0.0F, next_position), true) || !demuxer_[1]->seek(std::max(0.0F, next_position), true)) {
            // restore position if unable to perform forward seek
            std::cerr << "Unable to seek" << std::endl;
            demuxer_[0]->seek(std::max(0.0F, current_position), true);
            demuxer_[1]->seek(std::max(0.0F, current_position), true);
          };

          seeking_ = false;

          frame_queue_[0]->pop(frame_left);
          left_pts = frame_left->pts;
          left_previous_decoded_picture_number = -1;
          left_decoded_picture_number = 1;

          frame_queue_[1]->pop(frame_right);
          right_pts = frame_right->pts - right_time_shift;
          right_previous_decoded_picture_number = -1;
          right_decoded_picture_number = 1;

          left_frames.clear();
          right_frames.clear();

          current_position = frame_left->pts * AV_TIME_TO_SEC;

          videoControl->set_seeking(false);
        }
      } 

      bool store_frames = false;

      if (videoControl->get_quit()) {
        break;
      } else {
        bool adjusting = false;

        // use the delta between current and previous PTS as the tolerance which determines whether we have to adjust
        if ((left_pts < 0) || is_behind(left_pts, right_pts, delta_left_pts)) {
          adjusting = true;

          if (frame_queue_[0]->pop(frame_left)) {
            left_decoded_picture_number++;
          }
        }
        if ((right_pts < 0) || is_behind(right_pts, left_pts, delta_right_pts)) {
          adjusting = true;

          if (frame_queue_[1]->pop(frame_right)) {
            right_decoded_picture_number++;
          }
        }

        if (!adjusting && videoControl->get_play()) {
          if (!frame_queue_[0]->pop(frame_left) || !frame_queue_[1]->pop(frame_right)) {
            if (frame_left != nullptr) {
              left_decoded_picture_number++;
            }
            if (frame_right != nullptr) {
              right_decoded_picture_number++;
            }

            timer_->update();
          } else {
            left_decoded_picture_number++;
            right_decoded_picture_number++;

            store_frames = true;

            if (frame_number > 0) {
              const int64_t frame_delay = frame_left->pts - left_pts;
              timer_->wait(frame_delay);
            } else {
              timer_->update();
            }
          }
        } else {
          timer_->update();
        }
      }
    

      if (frame_left != nullptr) {
        if ((left_decoded_picture_number - left_previous_decoded_picture_number) == 1) {
          left_deque.push_back(frame_left->pts - left_pts);
          delta_left_pts = left_deque.average();
        }

        left_pts = frame_left->pts;
        left_previous_decoded_picture_number = left_decoded_picture_number;
      }
      if (frame_right != nullptr) {
        float new_right_pts = frame_right->pts - right_time_shift;

        if ((right_decoded_picture_number - right_previous_decoded_picture_number) == 1) {
          right_deque.push_back(new_right_pts - right_pts);
          delta_right_pts = right_deque.average();
        }

        right_pts = new_right_pts;
        right_previous_decoded_picture_number = right_decoded_picture_number;
      }

      if (store_frames) {
        if (left_frames.size() >= 50) {
          left_frames.pop_back();
        }
        if (right_frames.size() >= 50) {
          right_frames.pop_back();
        }

        left_frames.push_front(std::move(frame_left));
        right_frames.push_front(std::move(frame_right));
      } else {
        if (frame_left != nullptr) {
          if (!left_frames.empty()) {
            left_frames[0] = std::move(frame_left);
          } else {
            left_frames.push_front(std::move(frame_left));
          }
        }
        if (frame_right != nullptr) {
          if (!right_frames.empty()) {
            right_frames[0] = std::move(frame_right);
          } else {
            right_frames.push_front(std::move(frame_right));
          }
        }
      }

      frame_offset = std::min(std::max(0, frame_offset + videoControl->get_frame_offset_delta()), static_cast<int>(left_frames.size()) - 1);

      const std::string current_total_browsable = string_sprintf("%d/%d", frame_offset + 1, static_cast<int>(left_frames.size()));
      // std::cout << "current_total_browsable " << current_total_browsable << " offset " << frame_offset << std::endl;

      auto frame = std::make_unique<CombinedFrame>();
      frame->frame_offset = frame_number;
      frame->pts = left_frames[frame_offset]->pts;// * AV_TIME_TO_SEC;
      frame->original = left_frames[frame_offset]->frame;
      frame->processed = right_frames[frame_offset]->frame;

      std::cout << "decoded frame " << frame_number << " at " << frame->pts * AV_TIME_TO_SEC << "s " << left_frames[frame_offset]->pts << " pts " << "[frame queue]" << std::endl;
      combined_frame_queue.push(std::move(frame));      

      if (frame_queue_[0]->is_finished() || frame_queue_[1]->is_finished()) {
        break;
      }

    }
  } catch (...) {
    exception_ = std::current_exception();
  }

  if (exception_ == nullptr) {
    std::cout << "[Decoder] Finished decoding." << std::endl;
    combined_frame_queue.finished();
  }

  quit();
}

void TwoStreamDecoder::quit() {
  std::cout << "quit event in TwoStreamDecoder" << std::endl;
  // combined_frame_queue.quit();
  frame_queue_[0]->quit();
  packet_queue_[0]->quit();
  frame_queue_[1]->quit();
  packet_queue_[1]->quit();
}
