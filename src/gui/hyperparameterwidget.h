#pragma once
#include <QWidget>
#include <QSlider>
#include <QLabel>
#include <QGridLayout>
#include <QPushButton>
#include <QGraphicsDropShadowEffect>

#include "../stabilization/videostabilizer.h"


// Define an enumeration to represent the three states
enum class PlayingVideoType {
    Original,
    Stylized,
    Stabilized,
    Flowvis
};

class VideoTypeToggle;


class HyperParameterWidget : public QWidget
{
    Q_OBJECT

public:
    HyperParameterWidget(QWidget *parent = nullptr);
    void initValues(hyperParams *params);
    VideoTypeToggle *videoTypeToggle() { return m_videoTypeToggle; }

public slots:
    void onSliderValueChanged();

private:
    hyperParams *m_params;

    QSlider *m_alphaSlider;
    QSlider *m_betaSlider;
    QSlider *m_gammaSlider;
    QSlider *m_numIterSlider;
    QSlider *m_stepSizeSlider;
    QSlider *m_momFacSlider;

    QLabel *m_alphaLabel;
    QLabel *m_betaLabel;
    QLabel *m_gammaLabel;
    QLabel *m_numIterLabel;
    QLabel *m_stepSizeLabel;
    QLabel *m_momFacLabel;

    VideoTypeToggle *m_videoTypeToggle;
};

class VideoTypeToggle : public QWidget
{
    Q_OBJECT

public:
    VideoTypeToggle(QWidget *parent = nullptr);

    PlayingVideoType currentState() const { return m_currentState; }

public slots:
    void onStateButtonClicked();

signals:
    void stateChanged(PlayingVideoType newState);

private:
    QPushButton* m_originalButton;
    QPushButton* m_stylizedButton;
    QPushButton* m_stabilizedButton;
    QPushButton* m_flowVisButton;
    PlayingVideoType m_currentState;
    QGraphicsDropShadowEffect* m_dropshadoweffect;
};