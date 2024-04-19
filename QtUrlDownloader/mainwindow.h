#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class UrlDownloader;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public slots:
    void startSearch();
    void textDownloaded();
private:
    Ui::MainWindow *ui;
    UrlDownloader * m_urlDownloader;
};

#endif // MAINWINDOW_H
