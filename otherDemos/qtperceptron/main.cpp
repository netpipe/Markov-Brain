#include <QApplication>
#include <QMainWindow>
#include <QSplitter>
#include <QTextBrowser>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QTimer>
#include <QTextEdit>
#include <QLabel>
#include <QGroupBox>
#include <QStatusBar>
#include <QFile>
#include <QImage>
#include <QColor>
#include <QRegularExpression>
#include <QCoreApplication>
#include <QTextCursor>

#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>

// ============================================================================
// Perceptron Network (Custom RNN/MLP Hybrid with Backpropagation Through Time)
// ============================================================================
struct NeuralNet {
    const int VOCAB = 96; // ASCII 32..127
    const int CTX = 8;    // Context window
    const int EMB = 32;   // Embedding dimension
    const int HID = 64;   // Hidden layer size
    const int MEM = 16;   // Memory state size

    std::vector<float> E, W1, b1, W2, b2, Wm, Wi, bm;
    std::vector<float> mem;
    std::mt19937 rng{123};

    float randn() { std::normal_distribution<float> d(0, 0.1); return d(rng); }

    NeuralNet() {
        E.resize(VOCAB * EMB); for(auto& v:E) v=randn();
        W1.resize(HID * (CTX*EMB + MEM)); for(auto& v:W1) v=randn();
        b1.resize(HID);
        W2.resize(VOCAB * HID); for(auto& v:W2) v=randn();
        b2.resize(VOCAB);
        Wm.resize(MEM * MEM);
        Wi.resize(MEM * CTX*EMB);
        bm.resize(MEM);
        mem.resize(MEM, 0.0f);
    }

    int char_to_idx(char c) { return std::max(0, std::min(95, (int)c - 32)); }
    char idx_to_char(int i) { return (char)(i + 32); }

    struct State {
        std::vector<float> old_mem, new_mem, emb, h, out;
        std::vector<int> chars;
    };

    // Trains on a sequence of characters using BPTT
    void train_sequence(const std::string& text, float lr) {
        int len = text.size();
        if(len == 0) return;
        std::vector<State> states(len);
        std::vector<float> current_mem = mem;

        // 1. Forward Pass
        for(int t=0; t<len; ++t) {
            states[t].old_mem = current_mem;
            std::string ctx = "";
            for(int k=1; k<=CTX; ++k) {
                int idx = t - k;
                char c = (idx >= 0) ? text[idx] : ' '; // Fixed: using char ' '
                ctx = c + ctx;
            }
            states[t].chars.resize(CTX);
            states[t].emb.resize(CTX * EMB, 0.0f);
            int offset = 0;
            for(int i=0; i<CTX; ++i) {
                int c = char_to_idx(ctx[i]);
                states[t].chars[i] = c;
                for(int j=0; j<EMB; ++j) states[t].emb[offset++] = E[c * EMB + j];
            }
            states[t].h.resize(HID, 0.0f);
            int in_size = CTX * EMB + MEM;
            for(int i=0; i<HID; ++i) {
                float sum = b1[i];
                for(int j=0; j<CTX*EMB; ++j) sum += W1[i * in_size + j] * states[t].emb[j];
                for(int j=0; j<MEM; ++j) sum += W1[i * in_size + CTX*EMB + j] * current_mem[j];
                states[t].h[i] = std::max(0.0f, sum); // ReLU
            }
            states[t].new_mem.resize(MEM, 0.0f);
            for(int i=0; i<MEM; ++i) {
                float sum = bm[i];
                for(int j=0; j<MEM; ++j) sum += Wm[i * MEM + j] * current_mem[j];
                for(int j=0; j<CTX*EMB; ++j) sum += Wi[i * CTX*EMB + j] * states[t].emb[j];
                states[t].new_mem[i] = std::tanh(sum);
            }
            states[t].out.resize(VOCAB, 0.0f);
            float max_val = -1e9f;
            for(int i=0; i<VOCAB; ++i) {
                float sum = b2[i];
                for(int j=0; j<HID; ++j) sum += W2[i * HID + j] * states[t].h[j];
                states[t].out[i] = sum;
                if(sum > max_val) max_val = sum;
            }
            float sum_exp = 0.0f;
            for(int i=0; i<VOCAB; ++i) {
                states[t].out[i] = std::exp(states[t].out[i] - max_val);
                sum_exp += states[t].out[i];
            }
            for(int i=0; i<VOCAB; ++i) states[t].out[i] /= sum_exp;
            current_mem = states[t].new_mem;
        }
        mem = current_mem;

        // 2. Backward Pass (BPTT)
        std::vector<float> d_new_mem(MEM, 0.0f);
        for(int t=len-1; t>=0; --t) {
            State& s = states[t];
            char target = text[t];
            std::vector<float> d_out(VOCAB, 0.0f);
            for(int i=0; i<VOCAB; ++i) d_out[i] = s.out[i];
            d_out[char_to_idx(target)] -= 1.0f; // Cross-entropy derivative

            std::vector<float> d_h(HID, 0.0f);
            for(int i=0; i<VOCAB; ++i) {
                b2[i] -= lr * d_out[i];
                for(int j=0; j<HID; ++j) {
                    float old_w = W2[i * HID + j];
                    W2[i * HID + j] -= lr * d_out[i] * s.h[j];
                    d_h[j] += old_w * d_out[i];
                }
            }
            std::vector<float> d_z1(HID, 0.0f);
            for(int i=0; i<HID; ++i) d_z1[i] = (s.h[i] > 0) ? d_h[i] : 0.0f;

            std::vector<float> d_emb(CTX * EMB, 0.0f);
            std::vector<float> d_old_mem(MEM, 0.0f);
            int in_size = CTX * EMB + MEM;
            for(int i=0; i<HID; ++i) {
                b1[i] -= lr * d_z1[i];
                for(int j=0; j<CTX*EMB; ++j) {
                    float old_w = W1[i * in_size + j];
                    W1[i * in_size + j] -= lr * d_z1[i] * s.emb[j];
                    d_emb[j] += old_w * d_z1[i];
                }
                for(int j=0; j<MEM; ++j) {
                    float old_w = W1[i * in_size + CTX*EMB + j];
                    W1[i * in_size + CTX*EMB + j] -= lr * d_z1[i] * s.old_mem[j];
                    d_old_mem[j] += old_w * d_z1[i];
                }
            }
            std::vector<float> d_pre_tanh(MEM, 0.0f);
            for(int i=0; i<MEM; ++i) {
                float dtanh = 1.0f - s.new_mem[i] * s.new_mem[i];
                d_pre_tanh[i] = d_new_mem[i] * dtanh;
                bm[i] -= lr * d_pre_tanh[i];
                for(int j=0; j<MEM; ++j) {
                    float old_w = Wm[i * MEM + j];
                    Wm[i * MEM + j] -= lr * d_pre_tanh[i] * s.old_mem[j];
                    d_old_mem[j] += old_w * d_pre_tanh[i];
                }
                for(int j=0; j<CTX*EMB; ++j) {
                    float old_w = Wi[i * CTX*EMB + j];
                    Wi[i * CTX*EMB + j] -= lr * d_pre_tanh[i] * s.emb[j];
                    d_emb[j] += old_w * d_pre_tanh[i];
                }
            }
            int offset = 0;
            for(int i=0; i<CTX; ++i) {
                int c = s.chars[i];
                for(int j=0; j<EMB; ++j) E[c * EMB + j] -= lr * d_emb[offset++];
            }
            d_new_mem = d_old_mem;
        }
    }

    std::vector<float> forward(const std::string& ctx) {
        std::vector<float> emb(CTX * EMB, 0.0f);
        int offset = 0;
        for(int i=0; i<CTX; ++i) {
            int c = char_to_idx((i < ctx.size()) ? ctx[i] : ' ');
            for(int j=0; j<EMB; ++j) emb[offset++] = E[c * EMB + j];
        }
        std::vector<float> h(HID, 0.0f);
        int in_size = CTX * EMB + MEM;
        for(int i=0; i<HID; ++i) {
            float sum = b1[i];
            for(int j=0; j<CTX*EMB; ++j) sum += W1[i * in_size + j] * emb[j];
            for(int j=0; j<MEM; ++j) sum += W1[i * in_size + CTX*EMB + j] * mem[j];
            h[i] = std::max(0.0f, sum);
        }
        std::vector<float> new_mem(MEM, 0.0f);
        for(int i=0; i<MEM; ++i) {
            float sum = bm[i];
            for(int j=0; j<MEM; ++j) sum += Wm[i * MEM + j] * mem[j];
            for(int j=0; j<CTX*EMB; ++j) sum += Wi[i * CTX*EMB + j] * emb[j];
            new_mem[i] = std::tanh(sum);
        }
        mem = new_mem;
        std::vector<float> out(VOCAB, 0.0f);
        float max_val = -1e9f;
        for(int i=0; i<VOCAB; ++i) {
            float sum = b2[i];
            for(int j=0; j<HID; ++j) sum += W2[i * HID + j] * h[j];
            out[i] = sum;
            if(sum > max_val) max_val = sum;
        }
        float sum_exp = 0.0f;
        for(int i=0; i<VOCAB; ++i) { out[i] = std::exp(out[i] - max_val); sum_exp += out[i]; }
        for(int i=0; i<VOCAB; ++i) out[i] /= sum_exp;
        return out;
    }
};

// ============================================================================
// Brain Wrapper
// ============================================================================
class Brain : public QObject {
    Q_OBJECT
public:
    NeuralNet nn;
    QString last_generated;
    int total_trained = 0;
    std::mt19937 gen_rng{std::random_device{}()};

    Brain(QObject* parent = nullptr) : QObject(parent) {}

    QString extractPDFText(const QByteArray& data) {
        QString raw = QString::fromLatin1(data);
        QString text;
        QRegularExpression re(R"(\(([^)]*)\)\s*Tj|\[([^\]]*)\]\s*TJ)");
        auto it = re.globalMatch(raw);
        while(it.hasNext()) {
            auto match = it.next();
            text += match.captured(1) + match.captured(2) + " ";
        }
        if(text.trimmed().isEmpty()) {
            for(QChar c : raw) if(c.isPrint()) text += c;
        }
        return text;
    }

    QString processImage(const QString& filePath) {
        QImage img(filePath);
        if(img.isNull()) return "";
        img = img.convertToFormat(QImage::Format_Grayscale8);
        img = img.scaled(16, 16, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
        QString vision = "\nIMG[\n";
        for(int y=0; y<16; ++y) {
            for(int x=0; x<16; ++x) {
                int val = qGray(img.pixel(x, y));
                const char* ramp = " .:-=+*#%@";
                vision += ramp[val * 9 / 255];
            }
            vision += "\n";
        }
        vision += "]\n";
        return vision;
    }

    void learnData(const QString& data) {
        std::string s = data.toStdString();
        int chunk_size = 200; // Truncated BPTT for speed
        for(int i=0; i<s.length(); i+=chunk_size) {
            std::string sub = s.substr(i, std::min(chunk_size, (int)s.length() - i));
            if(sub.length() < 5) continue;
            nn.train_sequence(sub, 0.01f);
            total_trained++;
        }
    }

    QString generate(const QString& seed, int len) {
        std::string s = seed.toStdString();
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for(int i=0; i<len; ++i) {
            std::string ctx = "";
            for(int k=1; k<=nn.CTX; ++k) {
                int idx = s.size() - k;
                char c = (idx >= 0) ? s[idx] : ' '; // Fixed: using char ' '
                ctx = c + ctx;
            }
            auto probs = nn.forward(ctx);
            float r = dist(gen_rng);
            float cum = 0;
            int chosen = 0;
            for(int j=0; j<nn.VOCAB; ++j) {
                cum += probs[j];
                if(r <= cum) { chosen = j; break; }
            }
            s += nn.idx_to_char(chosen);
        }
        last_generated = QString::fromStdString(s);
        return last_generated;
    }

    void feedback(bool positive) {
        if(last_generated.isEmpty()) return;
        std::string s = last_generated.toStdString();
        if(positive) nn.train_sequence(s, 0.01f);
        else nn.train_sequence(s, -0.005f); // Unlearn via gradient ascent
    }
};

// ============================================================================
// Main Window GUI
// ============================================================================
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
        setWindowTitle("Perceptron Brain Demo — Read, See, Code");
        resize(1000, 700);

        QSplitter* split = new QSplitter(Qt::Horizontal);
        QWidget* left = new QWidget;
        QVBoxLayout* lv = new QVBoxLayout(left);
        lv->setContentsMargins(10,10,10,10);

        QGroupBox* grp1 = new QGroupBox("Inputs (PDF, JPG, TXT)");
        QVBoxLayout* v1 = new QVBoxLayout;
        QPushButton* btnPDF = new QPushButton("Read PDF...");
        QPushButton* btnJPG = new QPushButton("See JPG...");
        QPushButton* btnTXT = new QPushButton("Read TXT / C++...");
        v1->addWidget(btnPDF); v1->addWidget(btnJPG); v1->addWidget(btnTXT);
        grp1->setLayout(v1);
        lv->addWidget(grp1);

        QGroupBox* grp2 = new QGroupBox("Code Teacher");
        QVBoxLayout* v2 = new QVBoxLayout;
        teachEdit = new QTextEdit;
        teachEdit->setPlaceholderText("Paste C++ code here to teach the brain...");
        QPushButton* btnTeach = new QPushButton("Teach (50 Epochs)");
        QPushButton* btnBasic = new QPushButton("Teach Basics");
        v2->addWidget(teachEdit);
        v2->addWidget(btnTeach);
        v2->addWidget(btnBasic);
        grp2->setLayout(v2);
        lv->addWidget(grp2);

        QGroupBox* grp3 = new QGroupBox("Brain Stats");
        QVBoxLayout* v3 = new QVBoxLayout;
        statsLabel = new QLabel("Sequences Processed: 0\nArchitecture: RNN/MLP");
        statsLabel->setWordWrap(true);
        v3->addWidget(statsLabel);
        grp3->setLayout(v3);
        lv->addWidget(grp3);
        lv->addStretch();
        split->addWidget(left);

        QWidget* right = new QWidget;
        QVBoxLayout* rv = new QVBoxLayout(right);
        rv->setContentsMargins(10,10,10,10);
        chat = new QTextBrowser;
        chat->setOpenLinks(false);
        rv->addWidget(chat);

        QHBoxLayout* row = new QHBoxLayout;
        input = new QLineEdit;
        input->setPlaceholderText("Enter code seed (e.g., 'for(') or text...");
        QPushButton* btnGen = new QPushButton("Generate (200 chars)");
        QPushButton* btnUp = new QPushButton("👍");
        QPushButton* btnDown = new QPushButton("👎");
        row->addWidget(input); row->addWidget(btnGen); row->addWidget(btnUp); row->addWidget(btnDown);
        rv->addLayout(row);

        split->addWidget(right);
        split->setSizes({300, 700});
        setCentralWidget(split);

        connect(btnPDF, &QPushButton::clicked, this, &MainWindow::onPDF);
        connect(btnJPG, &QPushButton::clicked, this, &MainWindow::onJPG);
        connect(btnTXT, &QPushButton::clicked, this, &MainWindow::onTXT);
        connect(btnTeach, &QPushButton::clicked, this, &MainWindow::onTeach);
        connect(btnBasic, &QPushButton::clicked, this, &MainWindow::onBasic);
        connect(btnGen, &QPushButton::clicked, this, &MainWindow::onGen);
        connect(input, &QLineEdit::returnPressed, this, &MainWindow::onGen);
        connect(btnUp, &QPushButton::clicked, [this]{ brain->feedback(true); updateStats(); addMessage("Reinforced!", "System"); });
        connect(btnDown, &QPushButton::clicked, [this]{ brain->feedback(false); updateStats(); addMessage("Unlearned.", "System"); });

        brain = new Brain(this);

        qApp->setStyleSheet(
            "QMainWindow, QWidget { background:#1e1e1e; color:#d4d4d4; font-family:'Consolas','Courier New',monospace; font-size:13px; }"
            "QTextBrowser { background:#121212; border:1px solid #333; color:#4ec9b0; }"
            "QLineEdit, QTextEdit { background:#252526; border:1px solid #444; color:#d4d4d4; padding:5px; }"
            "QPushButton { background:#0e639c; color:white; border:none; padding:8px 16px; }"
            "QPushButton:hover { background:#1177bb; }"
            "QGroupBox { border:1px solid #444; margin-top:10px; }"
            "QGroupBox::title { color:#569cd6; subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }"
        );

        addMessage("Hello! I am a Perceptron with Memory. Teach me C++ code, show me JPGs, or let me read PDFs!", "Brain");
        onBasic(); // Pre-train so it doesn't just output garbage
    }

    void addMessage(const QString& msg, const QString& sender) {
        QString color = (sender == "You") ? "#569cd6" : (sender == "Brain" ? "#4ec9b0" : "#dcdcaa");
        QString html = QString("<p><b style='color:%1'>%2:</b> <span style='color:#d4d4d4'>%3</span></p>")
                            .arg(color, sender, msg.toHtmlEscaped());
        chat->append(html);
        QTextCursor c = chat->textCursor(); c.movePosition(QTextCursor::End); chat->setTextCursor(c);
    }

    void updateStats() { statsLabel->setText(QString("Sequences Processed: %1\nArchitecture: RNN/MLP").arg(brain->total_trained)); }

    void onPDF() {
        QString f = QFileDialog::getOpenFileName(this, "Read PDF", "", "PDF Files (*.pdf)");
        if(f.isEmpty()) return;
        QFile file(f); if(!file.open(QIODevice::ReadOnly)) return;
        QString text = brain->extractPDFText(file.readAll());
        if(!text.isEmpty()) { brain->learnData(text); addMessage("Read PDF! Extracted " + QString::number(text.length()) + " chars.", "System"); }
        else addMessage("Could not extract readable text from PDF.", "System");
        updateStats();
    }

    void onJPG() {
        QString f = QFileDialog::getOpenFileName(this, "See JPG", "", "Images (*.jpg *.jpeg *.png)");
        if(f.isEmpty()) return;
        QString vision = brain->processImage(f);
        if(!vision.isEmpty()) { brain->learnData(vision); addMessage("I can see the image!\n" + vision, "Brain"); }
        updateStats();
    }

    void onTXT() {
        QString f = QFileDialog::getOpenFileName(this, "Read TXT", "", "Text Files (*.txt *.cpp *.h)");
        if(f.isEmpty()) return;
        QFile file(f); if(!file.open(QIODevice::ReadOnly)) return;
        QString text = QString::fromUtf8(file.readAll());
        brain->learnData(text); addMessage("Read " + QString::number(text.length()) + " chars.", "System");
        updateStats();
    }

    void onTeach() {
        QString text = teachEdit->toPlainText(); if(text.isEmpty()) return;
        addMessage("Teaching code...", "System");
        for(int i=0; i<50; ++i) brain->learnData(text);
        addMessage("Taught 50 epochs!", "System"); updateStats();
    }

    void onBasic() {
        QStringList snippets = { "int main() { return 0; }\n", "for(int i=0; i<10; ++i) { cout << i; }\n", "if(x > 0) { return true; } else { return false; }\n", "#include <iostream>\n", "void swap(int& a, int& b) { int t=a; a=b; b=t; }\n" };
        QString all = snippets.join("");
        addMessage("Pre-training on basic C++ snippets...", "System");
        for(int i=0; i<100; ++i) brain->learnData(all);
        addMessage("Basics learned! Try typing `for(` and hitting Generate.", "Brain"); updateStats();
    }

    void onGen() {
        QString seed = input->text(); if(seed.isEmpty()) seed = "int main() {";
        addMessage(seed, "You");
        QString gen = brain->generate(seed, 200);
        addMessage(gen, "Brain");
    }

private:
    Brain* brain;
    QTextBrowser* chat;
    QLineEdit* input;
    QTextEdit* teachEdit;
    QLabel* statsLabel;
};

int main(int argc, char **argv) {
    QApplication app(argc, argv);
    MainWindow w;
    w.show();
    return app.exec();
}

#include "main.moc"
