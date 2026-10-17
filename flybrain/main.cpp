#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTabWidget>
#include <QTextEdit>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QSlider>
#include <QComboBox>
#include <QFileDialog>
#include <QImage>
#include <QPixmap>
#include <QTimer>
#include <QPainter>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsEllipseItem>
#include <QGraphicsLineItem>
#include <QRandomGenerator>
#include <QMessageBox>
#include <QGroupBox>
#include <QFormLayout>
#include <QMovie>
#include <QStatusBar>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <string>
#include <iostream>

//not tested yet

// ============================================================================
// CORE ENGINE: FlyCortex (Neuromorphic Mushroom Body Simulation)
// ============================================================================

class FlyCortex {
public:
    int D_in, N_kc, K_active, D_out;
    float alpha; // Recurrent memory decay (Liquid State)

    std::vector<std::vector<float>> W_in;  // Fixed sparse random projection
    std::vector<std::vector<float>> W_out; // Trainable MBON readout weights
    std::vector<std::vector<float>> W_rec; // Sparse recurrent synapses
    std::vector<float> S_prev;             // Previous state (Memory)
    std::vector<float> S_current;          // Current active state
    std::vector<float> Y_current;          // Output probabilities

    std::mt19937 rng;

    FlyCortex(int din=256, int nkc=2000, int kact=100, int dout=96, float alpha=0.5f)
        : D_in(din), N_kc(nkc), K_active(kact), D_out(dout), alpha(alpha), rng(42) {
        S_prev.assign(N_kc, 0.0f);
        S_current.assign(N_kc, 0.0f);
        Y_current.assign(D_out, 0.0f);
        initWeights();
    }

    void initWeights() {
        W_in.assign(N_kc, std::vector<float>(D_in, 0.0f));
        W_out.assign(D_out, std::vector<float>(N_kc, 0.0f));
        W_rec.assign(N_kc, std::vector<float>(N_kc, 0.0f));

        std::uniform_int_distribution<int> dist_in(0, D_in - 1);
        for (int i = 0; i < N_kc; ++i) {
            for (int j = 0; j < 10; ++j) {
                W_in[i][dist_in(rng)] = 1.0f;
            }
        }

        std::normal_distribution<float> normal(0.0f, 0.1f);
        for (int i = 0; i < D_out; ++i) {
            for (int j = 0; j < N_kc; ++j) {
                W_out[i][j] = normal(rng);
            }
        }

        for (int i = 0; i < N_kc; ++i) {
            for (int j = 0; j < 5; ++j) {
                int src = std::uniform_int_distribution<int>(0, N_kc - 1)(rng);
                W_rec[i][src] = 0.5f;
            }
        }
    }

    void resetMemory() {
        S_prev.assign(N_kc, 0.0f);
    }

    void forward(const std::vector<float>& X) {
        std::vector<float> kc_potential(N_kc, 0.0f);

        for (int i = 0; i < N_kc; ++i) {
            float sum = 0;
            for (int j = 0; j < D_in; ++j) sum += W_in[i][j] * X[j];
            kc_potential[i] = sum;
        }

        for (int i = 0; i < N_kc; ++i) {
            float rec_sum = 0;
            for (int j = 0; j < N_kc; ++j) rec_sum += W_rec[i][j] * S_prev[j];
            kc_potential[i] += alpha * rec_sum;
        }

        std::vector<std::pair<float, int>> sorted_kc(N_kc);
        for (int i = 0; i < N_kc; ++i) sorted_kc[i] = {kc_potential[i], i};

        std::partial_sort(sorted_kc.begin(), sorted_kc.begin() + K_active, sorted_kc.end(),
                          [](const std::pair<float,int>& a, const std::pair<float,int>& b){
                              return a.first > b.first;
                          });

        S_prev = S_current;
        S_current.assign(N_kc, 0.0f);
        for (int i = 0; i < K_active; ++i) {
            int idx = sorted_kc[i].second;
            S_current[idx] = sorted_kc[i].first;
        }

        Y_current.assign(D_out, 0.0f);
        for (int i = 0; i < D_out; ++i) {
            float sum = 0;
            for (int j = 0; j < N_kc; ++j) sum += W_out[i][j] * S_current[j];
            Y_current[i] = sum;
        }

        float max_val = *std::max_element(Y_current.begin(), Y_current.end());
        float sum_exp = 0;
        for (int i = 0; i < D_out; ++i) {
            Y_current[i] = std::exp(Y_current[i] - max_val);
            sum_exp += Y_current[i];
        }
        for (int i = 0; i < D_out; ++i) Y_current[i] /= sum_exp;
    }

    int train(const std::vector<float>& X, int target_class, float lr) {
        forward(X);
        int pred_class = std::distance(Y_current.begin(), std::max_element(Y_current.begin(), Y_current.end()));

        std::vector<float> grad_out(D_out);
        for (int i = 0; i < D_out; ++i) {
            grad_out[i] = Y_current[i] - (i == target_class ? 1.0f : 0.0f);
        }

        for (int i = 0; i < D_out; ++i) {
            for (int j = 0; j < N_kc; ++j) {
                if (S_current[j] > 0) {
                    W_out[i][j] -= lr * grad_out[i] * S_current[j];
                }
            }
        }
        return pred_class;
    }

    void addNeurons(int count) {
        int old_nkc = N_kc;
        N_kc += count;

        S_prev.resize(N_kc, 0.0f);
        S_current.resize(N_kc, 0.0f);
        W_in.resize(N_kc);
        W_out.resize(D_out, std::vector<float>(N_kc, 0.0f));
        W_rec.resize(N_kc, std::vector<float>(N_kc, 0.0f));

        for(int i=0; i<D_out; ++i) W_out[i].resize(N_kc, 0.0f);
        for(int i=0; i<old_nkc; ++i) W_rec[i].resize(N_kc, 0.0f);

        std::uniform_int_distribution<int> dist_in(0, D_in - 1);
        std::normal_distribution<float> normal(0.0f, 0.1f);

        for (int i = old_nkc; i < N_kc; ++i) {
            W_in[i].assign(D_in, 0.0f);
            for (int j = 0; j < 10; ++j) W_in[i][dist_in(rng)] = 1.0f;
            for (int j = 0; j < D_out; ++j) W_out[j][i] = normal(rng);

            W_rec[i].assign(N_kc, 0.0f);
            for (int j = 0; j < 5; ++j) {
                int src = std::uniform_int_distribution<int>(0, N_kc - 1)(rng);
                W_rec[i][src] = 0.5f;
            }
            for (int j = 0; j < old_nkc; ++j) {
                if (std::uniform_real_distribution<float>(0,1)(rng) < 0.01f) {
                    W_rec[j][i] = 0.5f;
                }
            }
        }
        K_active = std::max(1, (int)(K_active * (float)N_kc / old_nkc));
    }
};

// ============================================================================
// UI COMPONENTS
// ============================================================================

class BrainView : public QWidget {
    Q_OBJECT
public:
    BrainView(QWidget* parent = nullptr) : QWidget(parent) {
        setMinimumSize(300, 300);
        setStyleSheet("background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px;");
    }
    void updateState(const std::vector<float>& state) {
        currentState = state;
        update();
    }
protected:
    void paintEvent(QPaintEvent*) override {
        if (currentState.empty()) return;
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);

        int cols = std::ceil(std::sqrt(currentState.size()));
        if (cols == 0) return;
        float w = (float)width() / cols;
        float h = (float)height() / cols;

        for (int i = 0; i < currentState.size(); ++i) {
            int col = i % cols;
            int row = i / cols;
            float val = currentState[i];

            int intensity = std::min(255, (int)(val * 150));
            QColor c(20 + intensity, 50 + intensity, 100 + intensity/2);

            p.setPen(Qt::NoPen);
            p.setBrush(c);
            p.drawEllipse(QRectF(col*w + 1, row*h + 1, w - 2, h - 2));
        }
    }
private:
    std::vector<float> currentState;
};

class TopologyView : public QGraphicsView {
    Q_OBJECT
public:
    TopologyView(QWidget* parent = nullptr) : QGraphicsView(parent) {
        scene = new QGraphicsScene(this);
        setScene(scene);
        setRenderHint(QPainter::Antialiasing);
        setStyleSheet("background-color: #0d0d0d; border: 1px solid #333;");
        setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    }

    void updateTopology(const FlyCortex* brain) {
        scene->clear();
        if (!brain || brain->N_kc == 0) return;

        int w = width() - 20;
        int h = height() - 20;

        for(int i=0; i<std::min(10, brain->D_out); ++i) {
            float y = (h / 10.0f) * i + h/20.0f;
            scene->addEllipse(w - 20, y, 10, 10, QPen(Qt::red), QBrush(QColor("#d32f2f")));
        }

        int sample_size = std::min(200, brain->N_kc);
        for(int i=0; i<sample_size; ++i) {
            float x = w/2.0f;
            float y = (h / (float)sample_size) * i;
            bool active = brain->S_current[i] > 0;
            QColor c = active ? QColor("#4caf50") : QColor("#333333");

            scene->addEllipse(x, y, 4, 4, QPen(Qt::NoPen), QBrush(c));

            if (active) {
                for(int j=0; j<std::min(10, brain->D_out); ++j) {
                    float weight = brain->W_out[j][i];
                    if (std::abs(weight) > 0.05f) {
                        // FIX: Use RGBA integer constructor for QColor
                        QColor line_c = weight > 0 ? QColor(0, 188, 212, 100) : QColor(255, 152, 0, 100);
                        scene->addLine(x+4, y+2, w-20, (h/10.0f)*j + h/20.0f + 5, QPen(line_c, 1));
                    }
                }
            }
        }

        for(int i=0; i<std::min(50, brain->D_in); ++i) {
             float y = (h / 50.0f) * i;
             scene->addEllipse(10, y, 4, 4, QPen(Qt::NoPen), QBrush(QColor("#ffffff")));
        }
    }
private:
    QGraphicsScene* scene;
};

class MetricsPlotter : public QWidget {
    Q_OBJECT
public:
    std::vector<float> loss_history;
    MetricsPlotter(QWidget* parent = nullptr) : QWidget(parent) {
        setMinimumHeight(100);
        setStyleSheet("background-color: #1e1e1e; border: 1px solid #333; border-radius: 4px;");
    }
    void addLoss(float loss) {
        loss_history.push_back(loss);
        if (loss_history.size() > 100) loss_history.erase(loss_history.begin());
        update();
    }
protected:
    void paintEvent(QPaintEvent*) override {
        if (loss_history.empty()) return;
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        p.fillRect(rect(), QColor("#1e1e1e"));

        float max_l = *std::max_element(loss_history.begin(), loss_history.end());
        if (max_l == 0) max_l = 1.0f;

        p.setPen(QPen(QColor("#ff5252"), 2));
        float step = (float)width() / loss_history.size();
        for (size_t i=1; i<loss_history.size(); ++i) {
            float y1 = height() - (loss_history[i-1] / max_l) * height() * 0.8f;
            float y2 = height() - (loss_history[i] / max_l) * height() * 0.8f;
            p.drawLine((i-1)*step, y1, i*step, y2);
        }
    }
};

// ============================================================================
// MAIN APPLICATION WINDOW
// ============================================================================

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    FlyCortex* brain;
    BrainView* brainView;
    TopologyView* topView;
    MetricsPlotter* metricsPlot;
    QTextEdit* chat_history;
    QLineEdit* prompt_input;
    QLineEdit* img_label_input;
    QLabel* img_display;
    QLabel* vid_display;
    QLabel* vid_status;
    QTimer* gen_timer;
    QMovie* movie;
    std::string gen_text;
    int gen_step;
    std::vector<float> current_img_vec;
    QSlider* slider_kc;
    QSlider* slider_sparsity;
    QSlider* slider_alpha;

    MainWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
        setWindowTitle("FlyCortex: Advanced Neuromorphic Engine v2.0");
        resize(1300, 850);
        brain = new FlyCortex();
        movie = nullptr;

        QWidget* central = new QWidget(this);
        QHBoxLayout* main_layout = new QHBoxLayout(central);
        main_layout->setContentsMargins(10, 10, 10, 10);
        main_layout->setSpacing(10);

        // --- Left Panel ---
        QWidget* left_panel = new QWidget();
        QVBoxLayout* left_layout = new QVBoxLayout(left_panel);

        brainView = new BrainView(this);
        topView = new TopologyView(this);

        QGroupBox* param_box = new QGroupBox("Cortex Architecture");
        QFormLayout* param_layout = new QFormLayout(param_box);

        slider_kc = new QSlider(Qt::Horizontal); slider_kc->setRange(500, 5000); slider_kc->setValue(2000);
        QLabel* lbl_kc = new QLabel("2000");
        connect(slider_kc, &QSlider::valueChanged, lbl_kc, [lbl_kc](int v){ lbl_kc->setText(QString::number(v)); });

        slider_sparsity = new QSlider(Qt::Horizontal); slider_sparsity->setRange(1, 20); slider_sparsity->setValue(5);
        QLabel* lbl_sparsity = new QLabel("5%");
        connect(slider_sparsity, &QSlider::valueChanged, lbl_sparsity, [lbl_sparsity](int v){ lbl_sparsity->setText(QString::number(v) + "%"); });

        slider_alpha = new QSlider(Qt::Horizontal); slider_alpha->setRange(0, 100); slider_alpha->setValue(50);
        QLabel* lbl_alpha = new QLabel("0.50");
        connect(slider_alpha, &QSlider::valueChanged, lbl_alpha, [lbl_alpha](int v){ lbl_alpha->setText(QString::number(v/100.0, 'f', 2)); });

        connect(slider_kc, &QSlider::sliderReleased, this, &MainWindow::rebuildBrain);
        connect(slider_sparsity, &QSlider::sliderReleased, this, &MainWindow::rebuildBrain);
        connect(slider_alpha, &QSlider::sliderReleased, this, &MainWindow::rebuildBrain);

        QPushButton* btn_neurogenesis = new QPushButton("Trigger Neurogenesis (+500 KCs)");
        btn_neurogenesis->setStyleSheet("background-color: #4caf50; color: white; font-weight: bold;");
        connect(btn_neurogenesis, &QPushButton::clicked, this, [this]() {
            brain->addNeurons(500);
            brainView->updateState(brain->S_current);
            topView->updateTopology(brain);
            statusBar()->showMessage("Added 500 new Kenyon Cells! Total: " + QString::number(brain->N_kc));
        });

        param_layout->addRow("Neurons (KC):", slider_kc); param_layout->addRow(lbl_kc);
        param_layout->addRow("Sparsity (K):", slider_sparsity); param_layout->addRow(lbl_sparsity);
        param_layout->addRow("Memory (Alpha):", slider_alpha); param_layout->addRow(lbl_alpha);
        param_layout->addRow(btn_neurogenesis);

        metricsPlot = new MetricsPlotter(this);

        left_layout->addWidget(new QLabel("KC Activation Map:"));
        left_layout->addWidget(brainView);
        left_layout->addWidget(new QLabel("Network Topology:"));
        left_layout->addWidget(topView);
        left_layout->addWidget(param_box);
        left_layout->addWidget(new QLabel("Training Loss:"));
        left_layout->addWidget(metricsPlot);

        // --- Center Panel ---
        QTabWidget* tabs = new QTabWidget();

        // Text Tab
        QWidget* text_tab = new QWidget();
        QVBoxLayout* text_layout = new QVBoxLayout(text_tab);
        QComboBox* corpus_combo = new QComboBox();
        corpus_combo->addItems({"Shakespeare", "Sci-Fi Cyberpunk", "C++ Code"});
        QPushButton* btn_learn = new QPushButton("Train Brain on Corpus");
        btn_learn->setStyleSheet("background-color: #0078d7; color: white;");

        connect(btn_learn, &QPushButton::clicked, [this, corpus_combo]() {
            std::string corpus = "";
            if(corpus_combo->currentIndex() == 0) corpus = "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune...";
            else if(corpus_combo->currentIndex() == 1) corpus = "The sky was a crisp, neon pink. The AI core hummed with the data of a billion galaxies. Space travel was no longer a dream, but a daily commute...";
            else corpus = "int main() { std::vector<int> v = {1, 2, 3}; for (int i : v) { std::cout << i << ' '; } return 0; }";

            statusBar()->showMessage("Training on Corpus...");
            QApplication::processEvents();
            metricsPlot->loss_history.clear();

            // FIX: Cast corpus.length() to int to avoid unsigned/signed mismatch and underflow
            for(int i=0; i < (int)corpus.length()-1; ++i) {
                brain->resetMemory();
                float epoch_loss = 0;
                std::vector<float> X(256, 0.0f);
                int c = (int)corpus[i]; if (c >= 256) c = 255; X[c] = 1.0f;
                int target = (int)corpus[i+1] - 32; if (target < 0 || target >= 96) target = 0;
                brain->train(X, target, 0.05f);
                epoch_loss -= std::log(brain->Y_current[target] + 1e-9f);
                metricsPlot->addLoss(epoch_loss / corpus.length());
            }
            brainView->updateState(brain->S_current);
            topView->updateTopology(brain);
            statusBar()->showMessage("Training complete!");
        });

        prompt_input = new QLineEdit(); prompt_input->setPlaceholderText("Enter prompt to continue (e.g., 'The sky was')...");
        QPushButton* btn_generate = new QPushButton("Generate Sequence");
        connect(btn_generate, &QPushButton::clicked, this, &MainWindow::generateTextFromPrompt);

        text_layout->addWidget(new QLabel("Select Corpus and Train:"));
        text_layout->addWidget(corpus_combo); text_layout->addWidget(btn_learn);
        text_layout->addSpacing(20);
        text_layout->addWidget(new QLabel("Enter Prompt and Generate:"));
        text_layout->addWidget(prompt_input); text_layout->addWidget(btn_generate);
        text_layout->addStretch();
        tabs->addTab(text_tab, "Text / Chat");

        // Image Tab
        QWidget* img_tab = new QWidget();
        QVBoxLayout* img_layout = new QVBoxLayout(img_tab);
        img_display = new QLabel("No Image Loaded");
        img_display->setFixedSize(200, 200); img_display->setAlignment(Qt::AlignCenter);
        img_display->setStyleSheet("background-color: #222; border: 1px dashed #555;");
        img_display->setScaledContents(true);

        QPushButton* btn_load_img = new QPushButton("Load Image");
        connect(btn_load_img, &QPushButton::clicked, [this]() {
            QString file = QFileDialog::getOpenFileName(this, "Load Image", "", "Images (*.png *.jpg *.bmp)");
            if (!file.isEmpty()) {
                current_img_vec = loadAndFlatten(file);
                img_display->setPixmap(QPixmap(file).scaled(200, 200, Qt::KeepAspectRatio, Qt::SmoothTransformation));
                statusBar()->showMessage("Image loaded and flattened to 16x16 grid.");
            }
        });

        img_label_input = new QLineEdit(); img_label_input->setPlaceholderText("Text to associate (e.g., 'red apple')");
        QPushButton* btn_memorize = new QPushButton("Memorize Association");
        connect(btn_memorize, &QPushButton::clicked, [this]() {
            QString label = img_label_input->text();
            if (label.isEmpty() || current_img_vec.empty()) { statusBar()->showMessage("Load image and enter label!"); return; }

            brain->resetMemory();
            std::vector<float> X = current_img_vec;
            int target = (int)label.at(0).toLatin1() - 32; if (target < 0 || target >= 96) target = 0;
            brain->train(X, target, 0.1f);

            for(int i=0; i<label.length(); ++i) {
                int curr_c = (int)label.at(i).toLatin1();
                X.assign(256, 0.0f);
                for(int k=0; k<256; ++k) X[k] = current_img_vec[k];
                if (curr_c < 256) X[curr_c] += 1.0f;

                int next_target = 0;
                if (i+1 < label.length()) {
                    next_target = (int)label.at(i+1).toLatin1() - 32; if (next_target < 0 || next_target >= 96) next_target = 0;
                }
                brain->train(X, next_target, 0.1f);
            }
            statusBar()->showMessage("Memorized image-text association!");
            topView->updateTopology(brain);
        });

        QPushButton* btn_recognize = new QPushButton("Recognize / Hallucinate");
        connect(btn_recognize, &QPushButton::clicked, [this]() {
            if (current_img_vec.empty()) { statusBar()->showMessage("Load image first!"); return; }
            generateFromImage();
        });

        img_layout->addWidget(img_display, 0, Qt::AlignCenter);
        img_layout->addWidget(btn_load_img); img_layout->addWidget(img_label_input);
        img_layout->addWidget(btn_memorize); img_layout->addWidget(btn_recognize);
        img_layout->addStretch();
        tabs->addTab(img_tab, "Images / Vision");

        // Video Tab
        QWidget* vid_tab = new QWidget();
        QVBoxLayout* vid_layout = new QVBoxLayout(vid_tab);
        vid_display = new QLabel("No Video (GIF) Loaded");
        vid_display->setFixedSize(200, 200); vid_display->setScaledContents(true);
        vid_display->setStyleSheet("background-color: #222; border: 1px dashed #555;");
        vid_status = new QLabel("Status: Idle");

        // FIX: Removed vid_display and vid_status from capture list, they are class member variables
        QPushButton* btn_load_vid = new QPushButton("Load GIF Sequence");
        connect(btn_load_vid, &QPushButton::clicked, [this]() {
            QString file = QFileDialog::getOpenFileName(this, "Load GIF", "", "GIF Files (*.gif)");
            if (!file.isEmpty()) {
                if (movie) movie->deleteLater();
                movie = new QMovie(file, QByteArray(), this);
                vid_display->setMovie(movie);
                movie->start();
                vid_status->setText("Status: Playing & Encoding...");

                // FIX: Removed movie and vid_status from capture list, they are class member variables
                connect(movie, &QMovie::frameChanged, this, [this]() {
                    QImage frame = movie->currentImage();
                    if (frame.isNull()) return;
                    QImage scaled = frame.scaled(16, 16, Qt::IgnoreAspectRatio, Qt::SmoothTransformation).convertToFormat(QImage::Format_Grayscale8);
                    std::vector<float> imgVec(256, 0.0f);
                    for(int y=0; y<16; ++y) {
                        for(int x=0; x<16; ++x) {
                            imgVec[y*16+x] = scaled.pixelColor(x, y).red() / 255.0f;
                        }
                    }
                    brain->forward(imgVec);
                    brainView->updateState(brain->S_current);
                    topView->updateTopology(brain);
                });
            }
        });

        vid_layout->addWidget(vid_display, 0, Qt::AlignCenter);
        vid_layout->addWidget(btn_load_vid); vid_layout->addWidget(vid_status);
        vid_layout->addStretch();
        tabs->addTab(vid_tab, "Video / Temporal");

        // --- Right Panel ---
        QWidget* right_panel = new QWidget();
        QVBoxLayout* right_layout = new QVBoxLayout(right_panel);
        right_layout->addWidget(new QLabel("Neural Activity Log & Generation:"));
        chat_history = new QTextEdit(); chat_history->setReadOnly(true);
        chat_history->setStyleSheet("background-color: #0d0d0d; color: #4caf50; font-family: 'Consolas', monospace;");
        right_layout->addWidget(chat_history);

        // Assembly
        main_layout->addWidget(left_panel, 3);
        main_layout->addWidget(tabs, 4);
        main_layout->addWidget(right_panel, 3);

        setCentralWidget(central);
        applyTheme();

        gen_timer = new QTimer(this);
        connect(gen_timer, &QTimer::timeout, this, &MainWindow::stepGeneration);

        statusBar()->showMessage("FlyCortex Initialized. Ready for training.");
        topView->updateTopology(brain);
    }

    ~MainWindow() { delete brain; if(movie) delete movie; }

    void applyTheme() {
        setStyleSheet("QMainWindow { background-color: #121212; } "
                      "QWidget { color: #e0e0e0; font-family: 'Segoe UI', Arial; } "
                      "QPushButton { background-color: #333; border: 1px solid #555; padding: 8px; border-radius: 4px; } "
                      "QPushButton:hover { background-color: #444; } QPushButton:pressed { background-color: #222; } "
                      "QLineEdit, QTextEdit, QComboBox { background-color: #222; border: 1px solid #444; padding: 5px; border-radius: 4px; } "
                      "QGroupBox { border: 1px solid #444; margin-top: 10px; } "
                      "QGroupBox::title { color: #e0e0e0; subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; } "
                      "QSlider::groove:horizontal { border: 1px solid #444; height: 4px; background: #333; margin: 2px 0; border-radius: 2px; } "
                      "QSlider::handle:horizontal { background: #0078d7; border: 1px solid #0078d7; width: 14px; margin: -6px 0; border-radius: 7px; }"
                      "QTabWidget::pane { border: 1px solid #444; } "
                      "QTabBar::tab { background: #222; color: #e0e0e0; padding: 8px 16px; border: 1px solid #444; border-bottom: none; } "
                      "QTabBar::tab:selected { background: #333; border-bottom: 2px solid #0078d7; }");
    }

    void rebuildBrain() {
        delete brain;
        int nkc = slider_kc->value();
        int kact = slider_sparsity->value() * nkc / 100;
        float alpha = slider_alpha->value() / 100.0f;
        brain = new FlyCortex(256, nkc, kact, 96, alpha);
        brainView->updateState(brain->S_current);
        topView->updateTopology(brain);
        statusBar()->showMessage("Brain rebuilt with " + QString::number(nkc) + " neurons.");
    }

    void generateTextFromPrompt() {
        gen_text = prompt_input->text().toStdString();
        brain->resetMemory();
        std::vector<float> X(256, 0.0f);
        for(char c : gen_text) {
            X.assign(256, 0.0f);
            if ((int)c < 256) X[(int)c] = 1.0f;
            brain->forward(X);
        }
        current_img_vec = X;
        gen_step = 0;
        chat_history->clear();
        chat_history->setPlainText(QString::fromStdString(gen_text));
        gen_timer->start(50);
    }

    void generateFromImage() {
        gen_text = "[Recognizing Image]: ";
        brain->resetMemory();
        gen_step = 0;
        chat_history->clear();
        chat_history->setPlainText(QString::fromStdString(gen_text));
        gen_timer->start(50);
    }

    void stepGeneration() {
        if (gen_step >= 100) { gen_timer->stop(); return; }
        brain->forward(current_img_vec);
        int pred = std::distance(brain->Y_current.begin(), std::max_element(brain->Y_current.begin(), brain->Y_current.end()));
        char c = (char)(pred + 32);
        gen_text += c;
        chat_history->setPlainText(QString::fromStdString(gen_text));

        current_img_vec.assign(256, 0.0f);
        if ((int)c < 256) current_img_vec[(int)c] = 1.0f;

        gen_step++;
        if (gen_step > 5 && (c == '\n' || c == '.' || c == ' ')) {
            gen_timer->stop();
        }
    }

    std::vector<float> loadAndFlatten(const QString& path) {
        QImage img(path);
        if (img.isNull()) return std::vector<float>(256, 0.0f);
        QImage scaled = img.scaled(16, 16, Qt::IgnoreAspectRatio, Qt::SmoothTransformation).convertToFormat(QImage::Format_Grayscale8);
        std::vector<float> vec(256, 0.0f);
        for(int y=0; y<16; ++y) {
            for(int x=0; x<16; ++x) {
                vec[y*16+x] = scaled.pixelColor(x, y).red() / 255.0f;
            }
        }
        return vec;
    }
};



int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    MainWindow w;
    w.show();
    return app.exec();
}

#include "main.moc"
