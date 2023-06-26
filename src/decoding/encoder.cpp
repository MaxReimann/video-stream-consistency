// Copyright (C) 2023 Max Reimann (max.reimann@hpi.de)
// licensed under GNU General Public License v2.0
#include "encoder.h"
#include <stdio.h>
#include "demuxer.h"
#include "video_decoder.h"


extern "C" {
    #include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
  #include <libavutil/opt.h>
  #include <libavformat/avio.h>
  #include <libavutil/imgutils.h> //for av_image_alloc only
}
#include "ffmpeg.h"


VideoEncoder::VideoEncoder(const std::string& inputVideoPath, const std::string& outputVideoPath)
: m_inputVideoPath(inputVideoPath)
, m_outputVideoPath(outputVideoPath)
, m_inputFormatCtx(nullptr)
, m_inputVideoStream(nullptr)
, m_outputFormatCtx(nullptr)
, m_outputVideoStream(nullptr) {
    // avcodec_register_all();
    // av_register_all();
    initialize();

   m_isRunning = true;
   workerThread = std::thread(&VideoEncoder::worker, this);

}

VideoEncoder::~VideoEncoder() {
    finalize();
}

void VideoEncoder::initialize() {

    Demuxer demux(m_inputVideoPath);
    VideoDecoder decoder(demux.video_codec_parameters());

    ffmpeg::check(avformat_alloc_output_context2(&m_outputFormatCtx, NULL, NULL, m_outputVideoPath.c_str()));
    /*
    * since input and output files are supposed to be identical (framerate, dimension, color format, ...)
    * we can safely set output codec values from first input file
    */
    m_outputVideoStream = avformat_new_stream(m_outputFormatCtx, NULL);
    if (!m_outputVideoStream) {
        throw ffmpeg::Error{"Could not allocate stream"};
    }

  const auto* const encoder = avcodec_find_encoder(demux.video_codec_parameters()->codec_id);
  if (encoder == nullptr) {
    throw ffmpeg::Error{"Unsupported video encoder codec " +  std::to_string(demux.video_codec_parameters()->codec_id)};
  }

   m_outputCodecContext = avcodec_alloc_context3(encoder);
    if (m_outputCodecContext == nullptr) {
        throw ffmpeg::Error{"Couldn't allocate video codec context"};
    }

    if (decoder.codec_context()->codec_type != AVMEDIA_TYPE_VIDEO) {
        throw ffmpeg::Error{"[VideoEncode] Expected video codec type in input video"};
    }
    

    // ffmpeg::check(avcodec_parameters_to_context(m_outputCodecContext, demux.video_codec_parameters())); // if we copy from demuxer, video has many grey blocks for some reason
    // m_outputCodecContext->time_base.num  = decoder.codec_context()->time_base.num;
    // m_outputCodecContext->time_base.den  = decoder.codec_context()->time_base.den;
    const AVRational microseconds = {1, AV_TIME_BASE};

    m_outputCodecContext->time_base  = microseconds; // demux.time_base() ;  //decoder.codec_context()->time_base doesn't work, demuxer->time_base gives much too high denominator, av_inv_q(decoder.codec_context()->framerate) gives 0
    // m_outputCodecContext->codec = encoder;
    // m_outputCodecContext->time_base.den  = decoder.codec_context()->time_base.den;
    m_outputCodecContext->width   = decoder.codec_context()->width;
    m_outputCodecContext->height  = decoder.codec_context()->height;

    /* take first format from list of supported formats */
    if (encoder->pix_fmts)
        m_outputCodecContext->pix_fmt = encoder->pix_fmts[0];
    else
        m_outputCodecContext->pix_fmt = decoder.codec_context()->pix_fmt;
    m_outputCodecContext->sample_aspect_ratio = decoder.codec_context()->sample_aspect_ratio;
    if (m_outputFormatCtx->oformat->flags & AVFMT_GLOBALHEADER)
        m_outputFormatCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    std::cout << "type" << m_outputCodecContext->codec_type << " id " << m_outputCodecContext->codec_id << std::endl;
    std::cout << "width" << m_outputCodecContext->width << " height " << m_outputCodecContext->height << std::endl;

    //H.264 specific options
    // m_outputCodecContext->gop_size = 25;
    // m_outputCodecContext->level = 31;
     ffmpeg::check(av_opt_set(m_outputCodecContext->priv_data, "crf", "18", 0));
     ffmpeg::check(av_opt_set(m_outputCodecContext->priv_data, "profile", "main", 0));
    //  ffmpeg::check(av_opt_set(m_outputCodecContext->priv_data, "preset", "fast", 0));

    std::cout << "time_base.num = " << m_outputCodecContext->time_base.num << " time_base.den = " << m_outputCodecContext->time_base.den << std::endl;
    std::cout << "type" << m_outputCodecContext->codec_type << " id " << m_outputCodecContext->codec_id << std::endl;
    
    ffmpeg::check(avcodec_open2(m_outputCodecContext, encoder, nullptr));


    ffmpeg::check(avcodec_parameters_from_context(m_outputVideoStream->codecpar, m_outputCodecContext));

    //* dump av format informations
    av_dump_format(m_outputFormatCtx, 0, m_outputVideoPath.c_str(), 1);

    ffmpeg::check(avio_open(&m_outputFormatCtx->pb, m_outputVideoPath.c_str(), AVIO_FLAG_WRITE));

    /* yes! this is redundant */
    // avformat_close_input(&m_inputFormatCtx);

    ffmpeg::check(avformat_write_header(m_outputFormatCtx, NULL));
}

// The worker thread method
void VideoEncoder::worker() {
    while (m_isRunning) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]{ return !images.empty() || !m_isRunning; });
        
        if (!m_isRunning && images.empty()) {
            break;
        }

        FramePts frame = images.front();
        images.pop();
        lock.unlock();

        encodeImage(frame.image, frame.pts);
    }
}

void VideoEncoder::addImage(const QImage& image, int64_t timestamp) {
    std::lock_guard<std::mutex> lock(mtx);
    images.push(FramePts{image.copy(), timestamp});
    cv.notify_one();
}


void VideoEncoder::finalPacket()
{
    int ret;
    AVFrame* frame = NULL; // sending NULL flushes the encoder
    AVPacket pkt = {0};
    av_init_packet(&pkt);

    ret = avcodec_send_frame(m_outputCodecContext, frame);
    if (ret < 0) {
        fprintf(stderr, "Error sending a frame for encoding\n");
        exit(1);
    }

    while (ret >= 0) {
        ret = avcodec_receive_packet(m_outputCodecContext, &pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            fprintf(stderr, "Error during encoding\n");
            exit(1);
        }

        av_packet_rescale_ts(&pkt, m_outputCodecContext->time_base, m_outputVideoStream->time_base);

        pkt.stream_index = m_outputVideoStream->index;
        av_interleaved_write_frame(m_outputFormatCtx, &pkt);
        av_packet_unref(&pkt);
    }

    av_frame_free(&frame);
}



void VideoEncoder::encodeImage(const QImage& image, int64_t timestamp) {
    // Convert QImage to AVFrame
    AVFrame* frame = av_frame_alloc();
    frame->format = m_outputCodecContext->pix_fmt;
    frame->width = m_outputCodecContext->width;
    frame->height = m_outputCodecContext->height;
    frame->pts = timestamp;// / 48.25;
    
    
    int ret = av_frame_get_buffer(frame, 0);
    if (ret < 0) {
        // Handle error
        throw std::runtime_error("Failed to allocate buffer for frame, error: " + std::to_string(ret));
    }

    // Assuming QImage format is RGB888
    QImage rgbImage = image.convertToFormat(QImage::Format_RGB888);
    
    uint8_t* inData[1] = { rgbImage.bits() }; // RGB24 has one plane
    int inLinesize[1] = { 3 * rgbImage.width() }; // RGB stride
    
    SwsContext* swsContext = sws_getContext(rgbImage.width(), rgbImage.height(), AV_PIX_FMT_RGB24, 
                                             frame->width, frame->height, static_cast<AVPixelFormat>(frame->format), 
                                             0, NULL, NULL, NULL);
    
    sws_scale(swsContext, inData, inLinesize, 0, rgbImage.height(), frame->data, frame->linesize);
    
    sws_freeContext(swsContext);

    // Encode the image into the output video
    AVPacket pkt = {0};
    av_init_packet(&pkt);

    ret = avcodec_send_frame(m_outputCodecContext, frame);

    while (ret >= 0) {
        ret = avcodec_receive_packet(m_outputCodecContext, &pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            // Handle error. TODO: actually handle error, there could be ones where we can recover
            throw std::runtime_error("Failed to send frame for encoding, error: " + std::to_string(ret));
        }
        /* rescale output packet timestamp values from codec to stream timebase */
        av_packet_rescale_ts(&pkt, m_outputCodecContext->time_base, m_outputVideoStream->time_base);

        pkt.stream_index = m_outputVideoStream->index;
        av_interleaved_write_frame(m_outputFormatCtx, &pkt);
        av_packet_unref(&pkt);
    }

    av_frame_free(&frame);
}

void VideoEncoder::finalize() {
    std::lock_guard<std::mutex> lock(mtx);
    m_isRunning = false;
    cv.notify_one();
    workerThread.join();
    while (!images.empty()) {
        encodeImage(images.front().image, images.front().pts);
        images.pop();
    }

    finalPacket();
    av_write_trailer(m_outputFormatCtx);

    avcodec_close(m_outputCodecContext);
    av_freep(&m_outputCodecContext);
    av_freep(&m_outputFormatCtx->streams[0]);

    avio_close(m_outputFormatCtx->pb);
    av_free(m_outputFormatCtx);
}
