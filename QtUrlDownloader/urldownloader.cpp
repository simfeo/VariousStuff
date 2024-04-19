#include "urldownloader.h"
#include <QAuthenticator>

UrlDownloader::UrlDownloader(QUrl url, QObject *parent) :
    QObject(parent)
{
    connect(&m_WebCtrl, SIGNAL (finished(QNetworkReply*)), this, SLOT (fileDownloaded(QNetworkReply*)) );

    QNetworkRequest request(url);
    m_WebCtrl.get(request);
}

UrlDownloader::~UrlDownloader()
{
}

void UrlDownloader::fileDownloaded(QNetworkReply* pReply) {
    m_DownloadedData = pReply->readAll();
    //emit a signal
    pReply->deleteLater();
    emit signalDownloaded();
}

QByteArray UrlDownloader::downloadedData() const {
    return m_DownloadedData;
}


