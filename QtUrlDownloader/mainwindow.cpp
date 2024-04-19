#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "urldownloader.h"

#include <QTextCodec>


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    m_urlDownloader = nullptr;
    connect(ui->buttonStart, SIGNAL(released()),this, SLOT(startSearch()));
    ui->inputUrl->setValidator(new QRegExpValidator(QRegExp("^http://.*[\\w|/]$")));
    ui->maximumThreads->setValidator(new QIntValidator(0,200, this));
    ui->maximumUrlCount->setValidator(new QIntValidator(0,20000, this));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::startSearch()
{
    QUrl inputUrl = ui->inputUrl->text();
    m_urlDownloader = new UrlDownloader(inputUrl,this);
    connect(m_urlDownloader,SIGNAL(signalDownloaded()),this,SLOT(textDownloaded()));

}

void MainWindow::textDownloaded()
{
    QString data (m_urlDownloader->downloadedData().toStdString().c_str());
    qDebug("job is done");
}
