#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QFileDialog>
#include <QPainter>
#include <QTimer>
#include <QGroupBox>
#include <QStatusBar>
#include <QThread>
#include <QProgressBar>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <zlib.h>
#include <QMessageBox>

//untested

//based on https://github.com/snedea/flybrain/tree/main

// ============================================================================
// CORE ENGINE: Real Fly Connectome Simulator
// ============================================================================

class BrainWorker : public QObject {
    Q_OBJECT
public:
    struct Connectome {
        uint32_t n_neurons = 0, n_edges = 0;
        std::vector<uint32_t> row_ptr;
        std::vector<uint32_t> col_idx;
        std::vector<float> weights;
        std::vector<uint8_t> region;
        std::vector<uint16_t> group;
        std::vector<float> V;
        std::vector<int> refractory;
        std::vector<uint32_t> group_sizes;
    } brain;

    std::atomic<uint64_t> stimuli_mask{0};
    std::atomic<bool> running{false};

    BrainWorker() {
        brain.group_sizes.assign(63, 0);
    }

public slots:
    void loadConnectome(const QString& path) {
        gzFile file = gzopen(path.toStdString().c_str(), "rb");
        if (!file) { emit loadFinished(false, "Failed to open file."); return; }

        gzread(file, &brain.n_neurons, sizeof(uint32_t));
        gzread(file, &brain.n_edges, sizeof(uint32_t));

        if (brain.n_neurons > 200000 || brain.n_edges > 10000000) {
            gzclose(file); emit loadFinished(false, "File size exceeds sanity limits."); return;
        }

        emit loadingProgress(10);

        #pragma pack(push, 1)
        struct EdgeDisk { uint32_t pre, post; float weight; };
        struct NeuronDisk { uint8_t region; uint16_t group; };
        #pragma pack(pop)

        std::vector<EdgeDisk> disk_edges(brain.n_edges);
        gzread(file, disk_edges.data(), brain.n_edges * sizeof(EdgeDisk));
        emit loadingProgress(30);

        // Build CSR
        brain.row_ptr.assign(brain.n_neurons + 1, 0);
        for (size_t i = 0; i < brain.n_edges; ++i) {
            if (disk_edges[i].pre < brain.n_neurons)
                brain.row_ptr[disk_edges[i].pre + 1]++;
        }
        for (uint32_t i = 0; i < brain.n_neurons; ++i) brain.row_ptr[i + 1] += brain.row_ptr[i];

        brain.col_idx.resize(brain.n_edges);
        brain.weights.resize(brain.n_edges);
        std::vector<uint32_t> current_ptr = brain.row_ptr;
        for (size_t i = 0; i < brain.n_edges; ++i) {
            uint32_t pre = disk_edges[i].pre;
            if (pre < brain.n_neurons) {
                uint32_t dest = current_ptr[pre]++;
                brain.col_idx[dest] = disk_edges[i].post;
                brain.weights[dest] = disk_edges[i].weight;
            }
        }
        emit loadingProgress(60);

        std::vector<NeuronDisk> disk_neurons(brain.n_neurons);
        gzread(file, disk_neurons.data(), brain.n_neurons * sizeof(NeuronDisk));

        brain.region.resize(brain.n_neurons);
        brain.group.resize(brain.n_neurons);
        brain.V.assign(brain.n_neurons, 0.0f);
        brain.refractory.assign(brain.n_neurons, 0);

        std::fill(brain.group_sizes.begin(), brain.group_sizes.end(), 0);
        for (uint32_t i = 0; i < brain.n_neurons; ++i) {
            brain.region[i] = disk_neurons[i].region;
            brain.group[i] = disk_neurons[i].group;
            if (brain.group[i] < 63) brain.group_sizes[brain.group[i]]++;
        }

        gzclose(file);
        emit loadingProgress(100);
        emit loadFinished(true, QString("Loaded %1 neurons, %2 edges.").arg(brain.n_neurons).arg(brain.n_edges));
    }

    void start() {
        running = true;
        while(running) {
            tick();
            QThread::msleep(16); // ~60 ticks per second
        }
    }

    void stop() { running = false; }
    void addStimulus(int g) { stimuli_mask.fetch_or(1ULL << g); }
    void removeStimulus(int g) { stimuli_mask.fetch_and(~(1ULL << g)); }
    void clearStimuli() { stimuli_mask.store(0); }

signals:
    void stateUpdated(const std::vector<uint32_t>& group_active, const std::vector<uint32_t>& group_sizes, float walk, float turn, float back);
    void loadingProgress(int percent);
    void loadFinished(bool success, const QString& msg);

private:
    void tick() {
        if (brain.n_neurons == 0) return;

        std::vector<bool> fired(brain.n_neurons, false);
        std::vector<float> new_V(brain.n_neurons, 0.0f);

        for (uint32_t i = 0; i < brain.n_neurons; ++i) {
            if (brain.refractory[i] == 0 && brain.V[i] > 10.0f) {
                fired[i] = true;
            }
        }

        for (uint32_t i = 0; i < brain.n_neurons; ++i) {
            if (brain.refractory[i] > 0) {
                brain.refractory[i]--;
                new_V[i] = 0.0f;
            } else {
                new_V[i] = brain.V[i] * 0.8f;
            }
        }

        for (uint32_t pre = 0; pre < brain.n_neurons; ++pre) {
            if (fired[pre]) {
                uint32_t start = brain.row_ptr[pre];
                uint32_t end = brain.row_ptr[pre+1];
                for (uint32_t idx = start; idx < end; ++idx) {
                    uint32_t post = brain.col_idx[idx];
                    if (post < brain.n_neurons) {
                        new_V[post] += brain.weights[idx];
                    }
                }
            }
        }

        uint64_t mask = stimuli_mask.load();
        for (int g = 0; g < 63; ++g) {
            if ((mask & (1ULL << g)) != 0) {
                for (uint32_t i = 0; i < brain.n_neurons; ++i) {
                    if (brain.group[i] == g) new_V[i] += 20.0f;
                }
            }
        }

        std::vector<uint32_t> group_active(63, 0);
        float walk_act = 0, turn_act = 0, back_act = 0;

        for (uint32_t i = 0; i < brain.n_neurons; ++i) {
            if (fired[i]) brain.refractory[i] = 3;
            brain.V[i] = new_V[i];

            if (brain.V[i] > 1.0f) {
                if (brain.group[i] < 63) group_active[brain.group[i]]++;
                if (brain.group[i] == 42) walk_act += brain.V[i];
                if (brain.group[i] == 44) turn_act += brain.V[i];
                if (brain.group[i] == 45) back_act += brain.V[i];
            }
        }
        emit stateUpdated(group_active, brain.group_sizes, walk_act, turn_act, back_act);
    }
};

// ============================================================================
// UI COMPONENTS
// ============================================================================

class GroupView : public QWidget {
    Q_OBJECT
public:
    GroupView(QWidget* parent = nullptr) : QWidget(parent) {
        setMinimumSize(400, 200);
        setStyleSheet("background-color: #121212; border: 1px solid #333;");
    }

    void updateState(const std::vector<uint32_t>& counts, const std::vector<uint32_t>& szs) {
        active_counts = counts; sizes = szs; update();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this); p.setRenderHint(QPainter::Antialiasing); p.fillRect(rect(), QColor("#121212"));
        if (sizes.empty()) { p.setPen(Qt::white); p.drawText(rect(), Qt::AlignCenter, "Connectome Not Loaded"); return; }

        std::vector<std::pair<int, int>> regions = {{0, 17}, {17, 37}, {37, 42}, {42, 59}, {59, 63}};
        std::vector<QColor> region_colors = {QColor("#4caf50"), QColor("#2196f3"), QColor("#ff9800"), QColor("#f44336"), QColor("#9e9e9e")};
        std::vector<QString> region_names = {"Sensory", "Central", "Drives", "Motor", "Generic"};

        int y_offset = 10, cell_w = 20, cell_h = 35, margin = 10;

        for (size_t r = 0; r < regions.size(); ++r) {
            int start = regions[r].first, end = regions[r].second;
            p.setPen(region_colors[r]); p.drawText(margin, y_offset, region_names[r]);
            y_offset += 15;

            for (int i = start; i < end; ++i) {
                int x = margin + (i - start) * (cell_w + 4);
                float pct = (sizes[i] > 0) ? (float)active_counts[i] / sizes[i] : 0.0f;

                QColor c = region_colors[r]; c.setAlphaF(std::min(1.0f, 0.1f + pct * 2.0f));
                p.fillRect(x, y_offset, cell_w, cell_h, c);
                p.setPen(Qt::white); p.drawRect(x, y_offset, cell_w, cell_h);
            }
            y_offset += cell_h + 15;
        }
    }
private:
    std::vector<uint32_t> active_counts, sizes;
};

class FlyView : public QWidget {
    Q_OBJECT
public:
    float x = 0, y = 0, angle = 0;
    std::vector<QPointF> trail;

    void updateMotor(float forward, float reverse, float turn) {
        float speed = (forward - reverse) * 0.05f;
        angle += turn * 0.002f;
        x += speed * cos(angle);
        y += speed * sin(angle);
        trail.push_back(QPointF(x, y));
        if (trail.size() > 500) trail.erase(trail.begin());
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this); p.setRenderHint(QPainter::Antialiasing); p.fillRect(rect(), QColor("#1a1a1a"));
        float cx = width() / 2.0f, cy = height() / 2.0f;

                p.setPen(QPen(QColor(0, 188, 212, 100), 2));
        for (size_t i = 1; i < trail.size(); ++i) {
            p.drawLine(cx + (trail[i-1].x() - x), cy + (trail[i-1].y() - y),
                       cx + (trail[i].x() - x), cy + (trail[i].y() - y));
        }

        p.save();
        p.translate(cx, cy); p.rotate(angle * 180.0 / M_PI + 90);
        p.setBrush(Qt::yellow); p.setPen(Qt::black);
        p.drawEllipse(-5, -10, 10, 20);
        p.setBrush(QColor(255, 255, 255, 150));
        p.drawEllipse(-15, -5, 12, 8); p.drawEllipse(3, -5, 12, 8);
        p.restore();
    }
};

// ============================================================================
// MAIN APPLICATION WINDOW
// ============================================================================

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    BrainWorker* worker;
    QThread* thread;
    GroupView* groupView;
    FlyView* flyView;
    QProgressBar* progressBar;
    QLabel* statusLabel;

    MainWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
        setWindowTitle("Real Fly Brain Simulator (snedea/flybrain)"); resize(1100, 700);

        thread = new QThread(this);
        worker = new BrainWorker();
        worker->moveToThread(thread);

        connect(thread, &QThread::finished, worker, &QObject::deleteLater);
        connect(this, &MainWindow::startSimulation, worker, &BrainWorker::start);
        connect(this, &MainWindow::stopSimulation, worker, &BrainWorker::stop);
        connect(this, &MainWindow::loadFile, worker, &BrainWorker::loadConnectome);
        connect(worker, &BrainWorker::stateUpdated, this, &MainWindow::onStateUpdated);
        connect(worker, &BrainWorker::loadFinished, this, &MainWindow::onLoadFinished);
        connect(worker, &BrainWorker::loadingProgress, this, &MainWindow::onLoadingProgress);

        thread->start();
        emit startSimulation();

        QWidget* central = new QWidget(this);
        QHBoxLayout* main_layout = new QHBoxLayout(central);

        QWidget* left_panel = new QWidget();
        QVBoxLayout* left_layout = new QVBoxLayout(left_panel);

        QGroupBox* brain_box = new QGroupBox("Real Fly Connectome Activity (63 Groups)");
        QVBoxLayout* brain_layout = new QVBoxLayout(brain_box);
        groupView = new GroupView();
        brain_layout->addWidget(groupView);
        left_layout->addWidget(brain_box);

        QGroupBox* ctrl_box = new QGroupBox("Sensory Stimuli");
        QVBoxLayout* ctrl_layout = new QVBoxLayout(ctrl_box);

        auto createStimBtn = [&](const QString& name, int group) {
            QPushButton* btn = new QPushButton(name);
            btn->setStyleSheet("background-color: #333; color: white;");
            connect(btn, &QPushButton::clicked, this, [this, group]() { applyStimulus(group); });
            ctrl_layout->addWidget(btn);
        };

        createStimBtn("Shine Light (VIS)", 0);
        createStimBtn("Place Food (OLF)", 6);
        createStimBtn("Touch Fly (MECH)", 10);
        createStimBtn("Warm Up (THERMO)", 14);

        QPushButton* btnClear = new QPushButton("Clear Stimuli");
        connect(btnClear, &QPushButton::clicked, worker, &BrainWorker::clearStimuli);
        ctrl_layout->addWidget(btnClear);
        left_layout->addWidget(ctrl_box);

        QPushButton* btnLoad = new QPushButton("Load connectome.bin.gz");
        btnLoad->setStyleSheet("background-color: #0078d7; color: white; font-weight: bold;");
        connect(btnLoad, &QPushButton::clicked, this, &MainWindow::loadConnectome);
        left_layout->addWidget(btnLoad);

        progressBar = new QProgressBar(); progressBar->setVisible(false);
        left_layout->addWidget(progressBar);

        QWidget* right_panel = new QWidget();
        QVBoxLayout* right_layout = new QVBoxLayout(right_panel);
        QGroupBox* motor_box = new QGroupBox("Emergent Behavior (Motor Output)");
        QVBoxLayout* motor_layout = new QVBoxLayout(motor_box);
        flyView = new FlyView();
        motor_layout->addWidget(flyView);
        statusLabel = new QLabel("Status: Idle (Load connectome to begin)");
        motor_layout->addWidget(statusLabel);
        right_layout->addWidget(motor_box);

        main_layout->addWidget(left_panel, 4);
        main_layout->addWidget(right_panel, 6);
        setCentralWidget(central);
        applyTheme();
    }

    ~MainWindow() {
        emit stopSimulation();
        thread->quit();
        thread->wait();
    }

    void applyTheme() {
        setStyleSheet("QMainWindow { background-color: #121212; } "
                      "QWidget { color: #e0e0e0; font-family: 'Segoe UI', Arial; } "
                      "QPushButton { background-color: #333; border: 1px solid #555; padding: 8px; border-radius: 4px; } "
                      "QPushButton:hover { background-color: #444; } "
                      "QGroupBox { border: 1px solid #444; margin-top: 10px; } "
                      "QGroupBox::title { color: #e0e0e0; subcontrol-origin: margin; left: 10px; padding: 0 3px; }");
    }

signals:
    void startSimulation();
    void stopSimulation();
    void loadFile(const QString& path);

private slots:
    void loadConnectome() {
        QString file = QFileDialog::getOpenFileName(this, "Load Connectome", "", "Binary Files (*.bin.gz *.bin);;All Files (*)");
        if (!file.isEmpty()) {
            emit loadFile(file);
            statusLabel->setText("Status: Loading Connectome...");
            progressBar->setVisible(true);
        }
    }

    void applyStimulus(int group) {
        worker->addStimulus(group);
        QTimer::singleShot(200, this, [this, group]() {
            worker->removeStimulus(group);
        });
    }

    void onStateUpdated(const std::vector<uint32_t>& active, const std::vector<uint32_t>& sizes, float walk, float turn, float back) {
        groupView->updateState(active, sizes);
        flyView->updateMotor(walk, back, turn);
    }

    void onLoadFinished(bool success, const QString& msg) {
        progressBar->setVisible(false);
        if (success) {
            statusLabel->setText("Status: " + msg);
            QMessageBox::information(this, "Loaded", msg + "\nClick the sensory buttons to stimulate the brain!");
        } else {
            statusLabel->setText("Status: Error - " + msg);
        }
    }

    void onLoadingProgress(int pct) {
        progressBar->setValue(pct);
    }
};

#include "main.moc"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    MainWindow w;
    w.show();
    return app.exec();
}
