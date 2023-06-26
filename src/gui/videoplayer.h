#ifndef VIDEOPLAYER_H
#define VIDEOPLAYER_H

#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <QSharedPointer>
#include <QWidget>

#include "../stabilization/stabilizestream.h"
#include "../decoding/video_control.h"

#include "videowidget.h"
#include "hyperparameterwidget.h"


QT_BEGIN_NAMESPACE
class QAbstractButton;
class QSlider;
QT_END_NAMESPACE

class VideoPlayer : public QWidget
{
    Q_OBJECT

public:
    VideoPlayer(QWidget *parent = 0);
    ~VideoPlayer();
    QSharedPointer<StreamStabilizer> streamStabilizer;


public slots:
    void openFile(QString videoPath1, QString videoPath2);
    void play();
    void onFrameAvailable(int64_t framePts, QSharedPointer<QImage> image);
    void closeEvent(QCloseEvent *event) override;
    void videoTypeChanged(PlayingVideoType videoType);

private slots:
    // void mediaStateChanged(QMediaPlayer::State state);
    // void positionChanged(qint64 position);
    void setPosition(int position);
    void durationChanged(qint64 duration);

private:
    void createStreamStabilizer(QString videoOriginal, QString videoStylized);

    void setupDisplayWithFirstFrame();

    // QMediaPlayer mediaPlayer;
    QAbstractButton *playButton;
    QSlider *positionSlider;
    QSharedPointer<VideoWidget>  videoWidget;
    bool started;
    
    VideoControl *video_control;
    std::thread decoding_thread;
    QSharedPointer<HyperParameterWidget> hyperParameterWidget;
    
};

#endif // VIDEOPLAYER_H
