#include "hyperparameterwidget.h"
#include <QWidget>

#include <QSlider>
#include <QLabel>
#include <QGridLayout>


HyperParameterWidget::HyperParameterWidget(QWidget *parent) : QWidget(parent), m_params(nullptr)
{
    // Initial values (will be overwritten by the video stabilizer)
   float alpha = 6800.0f; 
   float beta = 6800.0f;
   float gamma = 2.0f;// make it high to increase consistency. however to remove ghosting artiffacts due to fast motion... reduce it to around 0.1
   float pyramidLevels = 2;
   int numIter = 150;
   float stepSize = 0.15f;
   float momFac = 0.15f;


    // Initialize the sliders
    m_alphaSlider = new QSlider(Qt::Horizontal);
    m_alphaSlider->setRange(0, 10000);
    m_alphaSlider->setValue(int(alpha)); //6800
    m_betaSlider = new QSlider(Qt::Horizontal);
    m_betaSlider->setRange(0, 10000);
    m_betaSlider->setValue(int(beta)); //6800
    m_gammaSlider = new QSlider(Qt::Horizontal);
    m_gammaSlider->setRange(0, 10000);
    m_gammaSlider->setValue(int(gamma * 1000) ); //2
    m_numIterSlider = new QSlider(Qt::Horizontal);
    m_numIterSlider->setRange(1, 1000);
    m_numIterSlider->setValue(numIter); //150
    m_stepSizeSlider = new QSlider(Qt::Horizontal);
    m_stepSizeSlider->setRange(0, 1000);
    m_stepSizeSlider->setValue(int(stepSize * 1000)); // 0.15
    m_momFacSlider = new QSlider(Qt::Horizontal);
    m_momFacSlider->setRange(0, 1000);
    m_momFacSlider->setValue(int(momFac * 1000)); // 0.15

    // Initialize the labels
    m_alphaLabel = new QLabel(QString::number(alpha));
    m_betaLabel = new QLabel(QString::number(beta));
    m_gammaLabel = new QLabel(QString::number(gamma));
    m_numIterLabel = new QLabel(QString::number(numIter));
    m_stepSizeLabel = new QLabel(QString::number(stepSize));
    m_momFacLabel = new QLabel(QString::number(momFac));

    // Initialize the slider name labels
    QLabel *alphaNameLabel = new QLabel("Alpha:");
    QLabel *betaNameLabel = new QLabel("Beta:");
    QLabel *gammaNameLabel = new QLabel("Gamma:");
    QLabel *numIterNameLabel = new QLabel("Num Iter:");
    QLabel *stepSizeNameLabel = new QLabel("Step Size:");
    QLabel *momFacNameLabel = new QLabel("Momentum Factor:");

    // Create the layout
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    // Add the parameter sliders widget
    QGridLayout *paramLayout = new QGridLayout;
    paramLayout->addWidget(numIterNameLabel, 0, 0);
    paramLayout->addWidget(m_numIterSlider, 0, 1);
    paramLayout->addWidget(m_numIterLabel, 0, 2);
    paramLayout->addWidget(stepSizeNameLabel, 1, 0);
    paramLayout->addWidget(m_stepSizeSlider, 1, 1);
    paramLayout->addWidget(m_stepSizeLabel, 1, 2);
    paramLayout->addWidget(alphaNameLabel, 2, 0);
    paramLayout->addWidget(m_alphaSlider, 2, 1);
    paramLayout->addWidget(m_alphaLabel, 2, 2);
    paramLayout->addWidget(betaNameLabel, 3, 0);
    paramLayout->addWidget(m_betaSlider, 3, 1);
    paramLayout->addWidget(m_betaLabel, 3, 2);
    paramLayout->addWidget(gammaNameLabel, 4, 0);
    paramLayout->addWidget(m_gammaSlider, 4, 1);
    paramLayout->addWidget(m_gammaLabel, 4, 2);
    paramLayout->addWidget(momFacNameLabel, 5, 0);
    paramLayout->addWidget(m_momFacSlider, 5, 1);
    paramLayout->addWidget(m_momFacLabel, 5, 2);


    QWidget *paramWidget = new QWidget(this);
    paramWidget->setLayout(paramLayout);
    mainLayout->addWidget(paramWidget);

    // Add the video type toggle widget
    m_videoTypeToggle = new VideoTypeToggle(this);
    mainLayout->addWidget(m_videoTypeToggle);

    setLayout(mainLayout);

    // Connect the signals and slots
    connect(m_alphaSlider, &QSlider::valueChanged, this, &HyperParameterWidget::onSliderValueChanged);
    connect(m_betaSlider, &QSlider::valueChanged, this, &HyperParameterWidget::onSliderValueChanged);
    connect(m_gammaSlider, &QSlider::valueChanged, this, &HyperParameterWidget::onSliderValueChanged);
    connect(m_numIterSlider, &QSlider::valueChanged, this, &HyperParameterWidget::onSliderValueChanged);
    connect(m_stepSizeSlider, &QSlider::valueChanged, this, &HyperParameterWidget::onSliderValueChanged);
    connect(m_momFacSlider, &QSlider::valueChanged, this, &HyperParameterWidget::onSliderValueChanged);

}

void HyperParameterWidget::initValues(hyperParams *params) {
    m_params = params;
    m_alphaSlider->setValue(int(m_params->alpha)); //6800
    m_betaSlider->setValue(int(m_params->beta)); //6800
    m_gammaSlider->setValue(int(m_params->gamma * 1000) ); //2
    m_numIterSlider->setValue(m_params->numIter); //150
    m_stepSizeSlider->setValue(int(m_params->stepSize * 1000)); // 0.15
    m_momFacSlider->setValue(int(m_params->momFac * 1000)); // 0.15
}

void HyperParameterWidget::onSliderValueChanged()
{
    m_params->alpha = float(m_alphaSlider->value());
    m_params->beta = float(m_betaSlider->value());
    m_params->gamma = float(m_gammaSlider->value()) / 1000.0f;
    m_params->numIter = m_numIterSlider->value();
    m_params->stepSize = float(m_stepSizeSlider->value()) / 1000.0f;
    m_params->momFac = float(m_momFacSlider->value()) / 1000.0f;

    m_alphaLabel->setText(QString::number(m_params->alpha));
    m_betaLabel->setText(QString::number(m_params->beta));
    m_gammaLabel->setText(QString::number(m_params->gamma));
    m_numIterLabel->setText(QString::number(m_params->numIter));
    m_stepSizeLabel->setText(QString::number(m_params->stepSize));
    m_momFacLabel->setText(QString::number(m_params->momFac));
    // emit hyperparametersChanged(m_alpha, m_beta, m_gamma, m_numIter, m_stepSize, m_momFac);
}


VideoTypeToggle::VideoTypeToggle(QWidget *parent) : QWidget(parent) {
    // Create three buttons to represent the three states
    m_originalButton = new QPushButton("Original", this);
    m_stylizedButton = new QPushButton("Stylized", this);
    m_stabilizedButton = new QPushButton("Stabilized", this);
    m_flowVisButton = new QPushButton("Optical Flow", this);

    // Set the initial state to original
    m_currentState = PlayingVideoType::Stabilized;
    m_stabilizedButton->setEnabled(false);
    m_stabilizedButton->setStyleSheet("QPushButton:disabled { border: 2px solid red; }");

    m_dropshadoweffect = new QGraphicsDropShadowEffect(this);
    m_dropshadoweffect->setBlurRadius(15);
    m_dropshadoweffect->setColor(QColor(255, 255, 255, 128));
    m_stylizedButton->setGraphicsEffect(m_dropshadoweffect);


    // Connect the buttons to the single slot that handles state changes
    connect(m_originalButton, &QPushButton::clicked, this, &VideoTypeToggle::onStateButtonClicked);
    connect(m_stylizedButton, &QPushButton::clicked, this, &VideoTypeToggle::onStateButtonClicked);
    connect(m_stabilizedButton, &QPushButton::clicked, this, &VideoTypeToggle::onStateButtonClicked);
    connect(m_flowVisButton, &QPushButton::clicked, this, &VideoTypeToggle::onStateButtonClicked);

    // Set the object name of each button to identify them in the slot
    m_originalButton->setObjectName("OriginalButton");
    m_stylizedButton->setObjectName("StylizedButton");
    m_stabilizedButton->setObjectName("StabilizedButton");
    m_flowVisButton->setObjectName("FlowVisButton");

    // Set up the layout for the buttons
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->addWidget(m_originalButton);
    buttonLayout->addWidget(m_stylizedButton);
    buttonLayout->addWidget(m_stabilizedButton);
    buttonLayout->addWidget(m_flowVisButton);
    setLayout(buttonLayout);
}

void VideoTypeToggle::onStateButtonClicked() {
    // Get the button that was clicked and set the state accordingly
    QPushButton* button = qobject_cast<QPushButton*>(sender());
    if (button) {
        if (button->objectName() == "OriginalButton") {
            m_currentState = PlayingVideoType::Original;
            m_originalButton->setEnabled(false);
            m_stylizedButton->setEnabled(true);
            m_stabilizedButton->setEnabled(true);
            m_flowVisButton->setEnabled(true);
        } else if (button->objectName() == "StylizedButton") {
            m_currentState = PlayingVideoType::Stylized;
            m_originalButton->setEnabled(true);
            m_stylizedButton->setEnabled(false);
            m_stabilizedButton->setEnabled(true);
            m_flowVisButton->setEnabled(true);
        } else if (button->objectName() == "StabilizedButton") {
            m_currentState = PlayingVideoType::Stabilized;
            m_originalButton->setEnabled(true);
            m_stylizedButton->setEnabled(true);
            m_stabilizedButton->setEnabled(false);
            m_flowVisButton->setEnabled(true);
        } else if (button->objectName() == "FlowVisButton") {
            m_currentState = PlayingVideoType::Flowvis;
            m_originalButton->setEnabled(true);
            m_stylizedButton->setEnabled(true);
            m_stabilizedButton->setEnabled(true);
        }


        button->setGraphicsEffect(m_dropshadoweffect);
        // Reset the style for the other buttons
        for (auto *btn : { m_originalButton, m_stylizedButton, m_stabilizedButton }) {
            if (btn != button) {
                btn->setStyleSheet("");
                btn->setGraphicsEffect(nullptr);
            }
        }

        // Set the style for the selected button
        button->setStyleSheet("QPushButton:disabled { border: 2px solid red; }");


        emit stateChanged(m_currentState);
    }
}