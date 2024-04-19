#ifndef URLDOWNLOADER_H
#define URLDOWNLOADER_H


#include <QObject>
#include <QByteArray>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>

class QAuthenticator;


class UrlDownloader : public QObject
{
    Q_OBJECT
public:
    explicit UrlDownloader(QUrl url, QObject *parent = 0);
    virtual ~UrlDownloader();
    QByteArray downloadedData() const;

signals:
    void signalDownloaded();

private slots:
    void fileDownloaded(QNetworkReply* pReply);
private:
    QNetworkAccessManager m_WebCtrl;
    QByteArray m_DownloadedData;
};
#endif // URLDOWNLOADER_H
