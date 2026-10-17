#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPainter>
#include <QLabel>
#include <QTimer>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>

//not tested yet

// --- The Fruit Fly Brain (FlyHash) ---
class FlyBrain {
public:
    int D_in, D_kc, D_out, K_active;
    std::vector<std::vector<float>> W1; // Fixed random projection (PN -> KC)
    std::vector<std::vector<float>> W2; // Trainable weights (KC -> MBON)

    FlyBrain(int d_in, int d_kc, int d_out, int k_active)
        : D_in(d_in), D_kc(d_kc), D_out(d_out), K_active(k_active) {
        
        W1.resize(D_kc, std::vector<float>(D_in, 0.0f));
        W2.resize(D_out, std::vector<float>(D_kc, 0.0f));

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist_in(0, D_in - 1);
        std::normal_distribution<float> normal(0.0f, 0.1f);
        
        // Initialize W1: Each Kenyon Cell connects to 6 random inputs (Sparse wiring)
        for (int i = 0; i < D_kc; ++i) {
            for (int j = 0; j < 6; ++j) {
                W1[i][dist_in(rng)] = 1.0f;
            }
        }
        
        // Initialize W2: Trainable readout weights
        for (int i = 0; i < D_out; ++i) {
            for (int j = 0; j < D_kc; ++j) {
                W2[i][j] = normal(rng);
            }
        }
    }

    // Returns {Output Probabilities, Sparse Kenyon Cell Activations}
    std::pair<std::vector<float>, std::vector<float>> forward(const std::vector<float>& x) {
        std::vector<float> kc(D_kc, 0.0f);
        // 1. Projection
        for (int i = 0; i < D_kc; ++i) {
            float sum = 0;
            for (int j = 0; j < D_in; ++j) if (W1[i][j] > 0) sum += W1[i][j] * x[j];
            kc[i] = sum;
        }

        // 2. Winner-Take-All (Sparsification) - Only top K active
        std::vector<std::pair<float, int>> kc_sorted(D_kc);
        for(int i=0; i<D_kc; ++i) kc_sorted[i] = {kc[i], i};
        std::sort(kc_sorted.begin(), kc_sorted.end(), std::greater<std::pair<float,int>>());
        
        std::vector<float> kc_sparse(D_kc, 0.0f);
        for (int i = 0; i < K_active; ++i) {
            int idx = kc_sorted[i].second;
            kc_sparse[idx] = kc_sorted[i].first;
        }

        // 3. MBON Output Layer
        std::vector<float> out(D_out, 0.0f);
        for (int i = 0; i < D_out; ++i)
            for (int j = 0; j < D_kc; ++j)
                out[i] += W2[i][j] * kc_sparse[j];
        
        // Softmax
        float max_out = *std::max_element(out.begin(), out.end());
        float sum_exp = 0;
        for(int i=0; i<D_out; ++i) {
            out[i] = std::exp(out[i] - max_out);
            sum_exp += out[i];
        }
        for(int i=0; i<D_out; ++i) out[i] /= sum_exp;

        return {out, kc_sparse};
    }

    void train(const std::vector<float>& x, int target_class, float lr) {
        auto res = forward(x);
        auto out = res.first;
        auto kc_sparse = res.second;
        
        std::vector<float> grad_out(D_out);
        for (int i = 0; i < D_out; ++i) {
            grad_out[i] = out[i] - (i == target_class ? 1.0f : 0.0f);
        }

        // Update only active Kenyon Cells (Hebbian-like plasticity)
        for (int i = 0; i < D_out; ++i) {
            for (int j = 0; j < D_kc; ++j) {
                if (kc_sparse[j] > 0) {
                    W2[i][j] -= lr * grad_out[i] * kc_sparse[j];
                }
            }
        }
    }
};

// --- GUI Visualization Classes ---

class NetworkView : public QWidget {
    Q_OBJECT
public:
    NetworkView(QWidget* parent = nullptr) : QWidget(parent) {
        setMinimumSize(400, 400);
        setStyleSheet("background-color: #1e1e1e; border: 1px solid #333;");
    }
    void setActivations(const std::vector<float>& kc_sparse, int D_kc) {
        activations = kc_sparse; this->D_kc = D_kc; update();
    }
protected:
    void paintEvent(QPaintEvent* event) override {
        QPainter p(this); p.setRenderHint(QPainter::Antialiasing);
        int cols = std::ceil(std::sqrt(D_kc));
        int cellW = width() / cols, cellH = height() / cols;
        for (int i = 0; i < D_kc; ++i) {
            int intensity = activations.empty() ? 0 : std::min(255, (int)(activations[i] * 500));
            p.fillRect((i % cols) * cellW, (i / cols) * cellH, cellW, cellH, 
                       QColor(50 + intensity, 50, 50 + intensity / 2));
        }
    }
private: std::vector<float> activations; int D_kc = 0;
};

class DatasetView : public QWidget {
    Q_OBJECT
public:
    DatasetView(QWidget* parent = nullptr) : QWidget(parent) {
        setMinimumSize(400, 400);
        setStyleSheet("background-color: #252526; border: 1px solid #333;");
        std::mt19937 rng(42); std::uniform_real_distribution<float> dist(0.0f, 1.0f); std::normal_distribution<float> noise(0.0f, 0.05f);
        for (int i = 0; i < 200; ++i) {
            float r = i < 100 ? 0.2f : 0.4f; float angle = dist(rng) * 2 * 3.14159f;
            dataset.push_back({{0.5f + r*std::cos(angle)+noise(rng), 0.5f + r*std::sin(angle)+noise(rng)}, i < 100 ? 0 : 1});
        }
    }
    const std::vector<std::pair<std::vector<float>, int>>& getDataset() const { return dataset; }
    void setPredictions(const std::vector<int>& preds) { predictions = preds; update(); }
protected:
    void paintEvent(QPaintEvent* event) override {
        QPainter p(this); p.setRenderHint(QPainter::Antialiasing); p.fillRect(rect(), QColor("#252526"));
        int margin = 20, plot_w = width() - 2*margin, plot_h = height() - 2*margin;
        for (size_t i = 0; i < dataset.size(); ++i) {
            int px = margin + dataset[i].first[0]*plot_w, py = margin + dataset[i].first[1]*plot_h;
            QColor c = (dataset[i].second == 0) ? QColor("#4ec9b0") : QColor("#c586c0");
            if (!predictions.empty() && predictions[i] != dataset[i].second) c = QColor("#f44747");
            p.setBrush(c); p.setPen(Qt::NoPen); p.drawEllipse(QPoint(px, py), 5, 5);
        }
    }
private: std::vector<std::pair<std::vector<float>, int>> dataset; std::vector<int> predictions;
};

// --- Main Application Window ---
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
        setWindowTitle("Fruit Fly Olfactory Brain - FlyHash Demo"); resize(850, 500);
        QWidget* central = new QWidget(this); QHBoxLayout* main_layout = new QHBoxLayout(central);
        dataset_view = new DatasetView(this); network_view = new NetworkView(this);
        QVBoxLayout* right_layout = new QVBoxLayout();
        
        lbl_stats = new QLabel("Epoch: 0 | Loss: N/A | Acc: N/A"); lbl_stats->setStyleSheet("color: white; font-weight: bold; font-size: 14px;");
        btn_train = new QPushButton("Train 1 Epoch"); btn_train->setStyleSheet("background-color: #0e639c; color: white; padding: 10px; font-size: 16px;");
        btn_reset = new QPushButton("Reset Brain"); btn_reset->setStyleSheet("background-color: #333; color: white; padding: 10px;");
        
        right_layout->addWidget(lbl_stats); right_layout->addWidget(dataset_view);
        right_layout->addWidget(btn_train); right_layout->addWidget(btn_reset);
        
        main_layout->addWidget(network_view); main_layout->addLayout(right_layout);
        setCentralWidget(central);
        
        brain = new FlyBrain(2, 500, 2, 25); // Input: 2D, 500 Kenyon Cells, 2 Classes, 25 active (5% sparsity)
        connect(btn_train, &QPushButton::clicked, this, &MainWindow::trainEpoch);
        connect(btn_reset, &QPushButton::clicked, this, &MainWindow::resetBrain);
    }

private slots:
    void trainEpoch() {
        float loss = 0; int correct = 0; const auto& ds = dataset_view->getDataset();
        for (const auto& pair : ds) {
            brain->train(pair.first, pair.second, 0.1f);
            auto out = brain->forward(pair.first).first;
            int pred = std::distance(out.begin(), std::max_element(out.begin(), out.end()));
            if (pred == pair.second) correct++;
            loss -= std::log(out[pair.second] + 1e-9f);
        }
        epoch++; current_loss = loss / ds.size(); current_acc = (float)correct / ds.size();
        
        std::vector<int> preds;
        for (const auto& pair : ds) {
            auto out = brain->forward(pair.first).first;
            preds.push_back(std::distance(out.begin(), std::max_element(out.begin(), out.end())));
        }
        dataset_view->setPredictions(preds);
        network_view->setActivations(brain->forward(ds[0].first).second, brain->D_kc);
        updateStats();
    }
    
    void resetBrain() {
        delete brain; brain = new FlyBrain(2, 500, 2, 25);
        epoch = 0; current_loss = 0; current_acc = 0;
        dataset_view->setPredictions({}); network_view->setActivations({}, brain->D_kc);
        updateStats();
    }
    
    void updateStats() {
        lbl_stats->setText(QString("Epoch: %1 | Loss: %2 | Acc: %3%")
            .arg(epoch).arg(current_loss, 0, 'f', 3).arg(current_acc * 100, 0, 'f', 1));
    }

private:
    FlyBrain* brain; DatasetView* dataset_view; NetworkView* network_view;
    QLabel* lbl_stats; QPushButton *btn_train, *btn_reset;
    int epoch = 0; float current_loss = 0, current_acc = 0;
};

#include "main.moc" // Necessary for Q_OBJECT in a single .cpp file

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    MainWindow w; w.show();
    return app.exec();
}