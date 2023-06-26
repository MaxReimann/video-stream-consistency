#include "videoplayer.h"
#include "videowidget.h"

#include <iostream>
#include <stdexcept>

#include <QtWidgets>
#include <QtConcurrent>
#include <qvideosurfaceformat.h>
#include <QProxyStyle>
#include "videoopenwidget.h"
#include "waiting_spinner.h"

#define AV_TIME_BASE 1000000


class JumpClickSliderStyle : public QProxyStyle
{
public:
    using QProxyStyle::QProxyStyle;

    int styleHint(QStyle::StyleHint hint, const QStyleOption* option = 0, const QWidget* widget = 0, QStyleHintReturn* returnData = 0) const
    {
        if (hint == QStyle::SH_Slider_AbsoluteSetButtons)
            return (Qt::LeftButton);
        return QProxyStyle::styleHint(hint, option, widget, returnData);
    }
};

class MyException : public QtConcurrent::Exception
{
public:
    MyException(const std::exception& err) : e(err), ptr(std::current_exception()) {}
    void raise() const { throw *this; }
    MyException* clone() const { return new MyException(*this); }
    std::exception error() const { return e; }
    std::exception_ptr errorPtr() const { return ptr; }
private:
    std::exception e;
    std::exception_ptr ptr;
};


VideoPlayer::VideoPlayer(QWidget *parent)
    : QWidget(parent)
    // , mediaPlayer(0, QMediaPlayer::VideoSurface)
    , playButton(0)
    , positionSlider(0), 
    started(false), video_control(nullptr)
{
    // VideoWidget *videoWidget = new VideoWidget;
    videoWidget = QSharedPointer<VideoWidget>(new VideoWidget);

    VideoOpenWidget *openWidget = new VideoOpenWidget(this);
    // openWidget->setModal(true);
    connect(openWidget, &VideoOpenWidget::videoStart,
            this, &VideoPlayer::openFile);


    playButton = new QPushButton;
    // playButton->setEnabled(false);
    playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));

    connect(playButton, &QAbstractButton::clicked,
            this, &VideoPlayer::play);

    positionSlider = new QSlider(Qt::Horizontal);
    positionSlider->setRange(0, 0);
    positionSlider->setValue(0);
    positionSlider->setStyle(new JumpClickSliderStyle(positionSlider->style()));

    connect(positionSlider, &QSlider::sliderMoved,
            this, &VideoPlayer::setPosition);

    QBoxLayout *controlLayout = new QHBoxLayout;
    controlLayout->setContentsMargins(0, 0, 0, 0);
    controlLayout->addWidget(playButton);
    controlLayout->addWidget(positionSlider);

    hyperParameterWidget = QSharedPointer<HyperParameterWidget>(new HyperParameterWidget(this));
    connect(hyperParameterWidget->videoTypeToggle(), &VideoTypeToggle::stateChanged,
            this, &VideoPlayer::videoTypeChanged);

    QBoxLayout *layout = new QVBoxLayout;
    layout->addWidget(videoWidget.get());
    layout->addLayout(controlLayout);
    layout->addWidget(hyperParameterWidget.get());

    setLayout(layout);

    qRegisterMetaType<int64_t>("int64_t");

    openWidget->show();
}

VideoPlayer::~VideoPlayer()
{
}


void VideoPlayer::createStreamStabilizer(QString videoOriginal, QString videoStylized) {
    auto demuxer = new Demuxer(videoOriginal.toStdString());
    auto decoder = new VideoDecoder(demuxer->video_codec_parameters());
    int width = decoder->width();
    int height = decoder->height();
    delete decoder;
    delete demuxer;
    int batchSize = 1;
    auto modelType = std::make_optional<QString>("pwcnet-light");

    auto ss = new StreamStabilizer(QDir(videoOriginal), QDir(videoStylized), std::nullopt, modelType, width, height, batchSize, true);
    streamStabilizer = QSharedPointer<StreamStabilizer>(ss);

    video_control = streamStabilizer->getVideoControl();
    hyperParameterWidget->initValues(streamStabilizer->getHyperParams());
    positionSlider->setRange(0, video_control->get_duration_in_milli_seconds());

    streamStabilizer->startBackgroundThread();
    bool success = this->streamStabilizer->processOneFrame(0);    // process one frame to display
}

void VideoPlayer::setupDisplayWithFirstFrame() {
    connect(streamStabilizer.get(), &StreamStabilizer::frameReady, this, &VideoPlayer::onFrameAvailable);

    // flush the first image that has already been processed to display
    while (streamStabilizer->getLastImage() == nullptr) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    onFrameAvailable(streamStabilizer->getLastPts(), streamStabilizer->getLastImage());

}

void VideoPlayer::openFile(QString videoOriginal, QString videoStylized)
{
    if (!videoOriginal.isEmpty() && !videoStylized.isEmpty()) {
        WaitingSpinnerWidget* spinner = new WaitingSpinnerWidget(Qt::ApplicationModal,this,true,false );//hyperParameterWidget.get(), true, true);
        spinner->start(); // starts spinning
        QtConcurrent::run([this, videoOriginal, videoStylized,spinner] {
            try {
                createStreamStabilizer(videoOriginal, videoStylized);
            } catch (const std::exception& e) {
                auto wrappedException = MyException(e);
                // Handle the exception
                QMetaObject::invokeMethod(QCoreApplication::instance(), [this, wrappedException, spinner] {
                    std::rethrow_exception(wrappedException.errorPtr());
                }, Qt::BlockingQueuedConnection);
            }

            QMetaObject::invokeMethod(QCoreApplication::instance(), [this, videoOriginal, videoStylized,spinner] {
                    setupDisplayWithFirstFrame();
                    // playButton->setEnabled(true);
                    spinner->stop();
            }, Qt::BlockingQueuedConnection);
        });
    }  else {
        playButton->setEnabled(false);
        std::cout << "Error: video file is empty" << std::endl;
    }

}

void VideoPlayer::play()
{

    if (video_control->get_play()) {
        if (started) {
            video_control->set_play(false);
            playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
        } 
    } else {
        video_control->set_play(true);
        playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPause));
    }

    if (!started) {
        started = true;
        std::cout << "Starting video" << std::endl;
        playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPause));

        QtConcurrent::run([this] {
            bool stop = false;
            bool *stopPtr = &stop;
            for (int i = 1;; i++) {

                QMetaObject::invokeMethod(QCoreApplication::instance(), [this, i, stopPtr] {
                    bool success = this->streamStabilizer->processOneFrame(i);    // we need to call this from the main thread
                    if (!success || this->video_control->get_quit()) {
                        *stopPtr = true;
                    }
                }, Qt::BlockingQueuedConnection);
                if (*stopPtr) {
                    std::cout << "Qt concurrent stopped" << std::endl;
                    break;
                }
            }
        });
    } else {
        std::cout << "Video already started." << std::endl;
    }
}

void VideoPlayer::durationChanged(qint64 duration)
{
    positionSlider->setRange(0, duration);
}

void VideoPlayer::setPosition(int position)
{

    if (video_control->get_is_seeking()) {
        return;
    }
    
    // if (video_control->get_current_position() == position) {
    //     return;
    // }

    video_control->set_seek_absolute(static_cast<int64_t>(position) * 1000);
    video_control->set_seeking(true);
}

void VideoPlayer::videoTypeChanged(PlayingVideoType videoType)
{
    if (videoType == PlayingVideoType::Original) {
        std::cout << "Video type changed to original" << std::endl;
        video_control->set_video_type(VideoType::ORIGINAL);
    } else if (videoType == PlayingVideoType::Stylized) {
        std::cout << "Video type changed to stylized" << std::endl;
        video_control->set_video_type(VideoType::STYLIZED);
    } else if (videoType == PlayingVideoType::Stabilized) {
        std::cout << "Video type changed to stabilized" << std::endl;
        video_control->set_video_type(VideoType::STABILIZED);
    } else if (videoType == PlayingVideoType::Flowvis) {
        std::cout << "Video type changed to stabilized stylized" << std::endl;
        video_control->set_video_type(VideoType::FLOWVIS);
    }
}

void VideoPlayer::onFrameAvailable(int64_t frame_pts, QSharedPointer<QImage> image)
{

    if (!videoWidget->videoSurface())
        return;
    auto surface = videoWidget->videoSurface();
    QVideoFrame frame(image->convertToFormat(QImage::Format_RGB32));

    if (!surface->isActive()) { 
        QVideoSurfaceFormat format(QSize(image->width(), image->height()), QVideoFrame::Format_RGB32, QAbstractVideoBuffer::NoHandle);
        surface->start(format);
    }


    surface->present(frame); // main thread
    video_control->set_current_position(frame_pts);

    if (!video_control->get_is_seeking()) {
        positionSlider->blockSignals(true);
        positionSlider->setValue( int(frame_pts / 1000));
        positionSlider->blockSignals(false);
    }
}

void VideoPlayer::closeEvent(QCloseEvent *event) {
    streamStabilizer->quit();

    // opportunity for background threads to close
    QThread::sleep(2);
    
    QWidget::closeEvent(event);
}

