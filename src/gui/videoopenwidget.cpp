#include "videoopenwidget.h"

#include <QtWidgets>


VideoOpenWidget::VideoOpenWidget(QWidget *parent) :  QDialog(parent)
{
    video1Label = new QLabel(tr("Original video:"));
    video2Label = new QLabel(tr("Stylized video:"));
    video1Edit = new QLineEdit;
    video2Edit = new QLineEdit;

    openButton1 = new QPushButton(tr("Open original video..."));
    connect(openButton1, &QAbstractButton::clicked, this, &VideoOpenWidget::openFile1);

    openButton2 = new QPushButton(tr("Open stylized video..."));
    connect(openButton2, &QAbstractButton::clicked, this, &VideoOpenWidget::openFile2);

    startButton = new QPushButton(tr("Start"));
    connect(startButton, &QAbstractButton::clicked, this, &VideoOpenWidget::start);
    startButton->setEnabled(false);

    QHBoxLayout *video1Layout = new QHBoxLayout;
    video1Layout->addWidget(video1Label);
    video1Layout->addWidget(video1Edit);

    QHBoxLayout *video2Layout = new QHBoxLayout;
    video2Layout->addWidget(video2Label);
    video2Layout->addWidget(video2Edit);

    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addLayout(video1Layout);
    mainLayout->addWidget(openButton1);
    mainLayout->addLayout(video2Layout);
    mainLayout->addWidget(openButton2);
    mainLayout->addWidget(startButton);

    setLayout(mainLayout);
}

void VideoOpenWidget::openFile1()
{
    QString videoFile = QFileDialog::getOpenFileName(this, tr("Open original video"), QString(), tr("Video Files (*.mp4 *.mov)"));

    if (!videoFile.isEmpty()) {
        video1Edit->setText(videoFile);
    }
    if (!video1Edit->text().isEmpty() && !video2Edit->text().isEmpty())
    {
        startButton->setEnabled(true);
    }
}

void VideoOpenWidget::openFile2()
{
    QString videoFile = QFileDialog::getOpenFileName(this, tr("Open stylized video"), QString(), tr("Video Files (*.mp4 *.mov)"));

    if (!videoFile.isEmpty()) {
        video2Edit->setText(videoFile);
    }

    if (!video1Edit->text().isEmpty() && !video2Edit->text().isEmpty())
    {
        startButton->setEnabled(true);
    }
}


void VideoOpenWidget::start()
{
    if (!video1Edit->text().isEmpty() && !video2Edit->text().isEmpty()) {
        emit videoStart(video1Edit->text(), video2Edit->text());
        this->close();
    } else {
        QMessageBox::warning(this, tr("Error"), tr("Please select both videos."));
    }
}
