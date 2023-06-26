#pragma once

#include <QWidget>
#include <QDialog>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>

// QT_BEGIN_NAMESPACE
// class QAbstractButton;
// QT_END_NAMESPACE

class VideoOpenWidget : public  QDialog
{
    Q_OBJECT

public:
    VideoOpenWidget(QWidget *parent = nullptr);

signals:
    void videoStart(QString, QString);

private slots:
    void openFile1();
    void openFile2();
    void start();
private:
    QLabel *video1Label;
    QLabel *video2Label;
    QLineEdit *video1Edit;
    QLineEdit *video2Edit;
    QPushButton *openButton1;
    QPushButton *openButton2;
    QPushButton *startButton;
};

