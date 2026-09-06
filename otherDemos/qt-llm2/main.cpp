#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTextEdit>
#include <QLineEdit>
#include <QPushButton>
#include <QFileDialog>
#include <QLabel>
#include <QFile>
#include <QTextStream>
#include <QRegularExpression>
#include <QMap>
#include <QVector>
#include <QSet>
#include <QDebug>
#include <QFileInfo>
#include <QtMath>
#include <QScrollBar>
#include <QMessageBox>
#include <algorithm>
#include <cmath>

// ============================================================================
// THE "BRAIN": TF-IDF Vector Search & Extractive QA Engine
// ============================================================================
class AgentBrain {
public:
    struct Chunk {
        int id;
        QString source;
        QString text;
        QMap<QString, int> termFreq;
        double magnitude;
    };

    // Ingest a book or text file
    void ingestFile(const QString& filePath) {
        QFile file(filePath);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) return;

        QString content = file.readAll();
        file.close();

        // Split by double newlines (paragraphs/chunks)
        QStringList paragraphs = content.split(QRegularExpression("\\n\\s*\\n"), QString::SkipEmptyParts);

        for (const QString& p : paragraphs) {
            if (p.trimmed().isEmpty()) continue;

            Chunk c;
            c.id = m_chunks.size();
            c.source = QFileInfo(filePath).fileName();
            c.text = p.trimmed();

            QStringList words = tokenize(c.text);
            QSet<QString> uniqueWords;

            for (const QString& w : words) {
                c.termFreq[w]++;
                uniqueWords.insert(w);
            }
            // Document frequency counts how many chunks contain a word
            for (const QString& w : uniqueWords) {
                m_docFreq[w]++;
            }
            m_chunks.append(c);
        }
    }

    // Compute Inverse Document Frequency (IDF) and chunk magnitudes
    void computeIDF() {
        m_idf.clear();
        int totalDocs = m_chunks.size();
        if (totalDocs == 0) return;

        for (auto it = m_docFreq.begin(); it != m_docFreq.end(); ++it) {
            // Smoothed IDF formula
            m_idf[it.key()] = log10(1.0 + (double)totalDocs / (double)it.value());
        }

        // Precompute vector magnitudes for fast Cosine Similarity
        for (Chunk& c : m_chunks) {
            double mag = 0;
            for (auto it = c.termFreq.begin(); it != c.termFreq.end(); ++it) {
                double tf = 1.0 + log10((double)it.value()); // Sublinear TF
                double w = tf * m_idf.value(it.key(), 0.0);
                mag += w * w;
            }
            c.magnitude = sqrt(mag);
        }
    }

    // "Think" and extract the best answer
    QString thinkAndAnswer(const QString& query) {
        QStringList queryWords = tokenize(query);
        if (queryWords.isEmpty()) return "I need more specific keywords to search my memory for.";

        // 1. Build Query Vector
        QMap<QString, double> queryVec;
        QMap<QString, int> qtf;
        for(const QString& w : queryWords) qtf[w]++;

        double queryMag = 0;
        for (auto it = qtf.begin(); it != qtf.end(); ++it) {
            double tf = 1.0 + log10((double)it.value());
            double w = tf * m_idf.value(it.key(), 0.0);
            queryVec[it.key()] = w;
            queryMag += w * w;
        }
        queryMag = sqrt(queryMag);

        if (queryMag == 0) return "Those concepts are not in my knowledge base.";

        // 2. Find Top Relevant Chunks (Cosine Similarity)
        struct SearchResult { int chunkId; double score; };
        QVector<SearchResult> chunkResults;

        for (const Chunk& c : m_chunks) {
            double dot = 0;
            for (auto it = queryVec.begin(); it != queryVec.end(); ++it) {
                double c_tf = c.termFreq.value(it.key(), 0);
                if (c_tf > 0) {
                    double c_w = (1.0 + log10((double)c_tf)) * m_idf.value(it.key(), 0.0);
                    dot += it.value() * c_w;
                }
            }
            double score = (c.magnitude > 0) ? (dot / (c.magnitude * queryMag)) : 0.0;
            if (score > 0.001) {
                chunkResults.append({c.id, score});
            }
        }

        std::sort(chunkResults.begin(), chunkResults.end(), [](const SearchResult& a, const SearchResult& b) {
            return a.score > b.score;
        });

        if (chunkResults.isEmpty()) return "I couldn't find any relevant information in my memory.";

        // 3. Extractive Summarization (Sentence Scoring)
        int topChunks = qMin(5, chunkResults.size());
        QVector<QPair<double, QString>> sentenceScores;

        for (int i = 0; i < topChunks; ++i) {
            const Chunk& c = m_chunks[chunkResults[i].chunkId];
            // Split chunk into sentences safely
            QStringList sentences = c.text.split(QRegularExpression("\\n+|(?<=[.!?])\\s+"), QString::SkipEmptyParts);

            for (const QString& sent : sentences) {
                if (sent.length() < 15) continue; // Skip tiny fragments

                QStringList sentWords = tokenize(sent);
                double sentScore = 0;
                int matchCount = 0;

                for (const QString& w : queryWords) {
                    if (sentWords.contains(w)) {
                        sentScore += m_idf.value(w, 0.0); // Weight rare words higher
                        matchCount++;
                    }
                }

                // Normalize by sentence length and add chunk relevance
                if (matchCount > 0) {
                    sentScore = (sentScore / sqrt(sentWords.size())) + chunkResults[i].score;
                    sentenceScores.append({sentScore, QString("[%1]\n%2").arg(c.source, sent)});
                }
            }
        }

        std::sort(sentenceScores.begin(), sentenceScores.end(), [](const QPair<double, QString>& a, const QPair<double, QString>& b) {
            return a.first > b.first;
        });

        if (sentenceScores.isEmpty()) return "I found relevant chapters, but couldn't extract a clear sentence answer.";

        int topSents = qMin(3, sentenceScores.size());
        QString response = "Here is the exact knowledge I synthesized from my memory:\n\n";
        for (int i = 0; i < topSents; ++i) {
            response += QString("%1\n\n").arg(sentenceScores[i].second);
        }

        return response;
    }

    void clear() {
        m_chunks.clear();
        m_docFreq.clear();
        m_idf.clear();
    }

    int chunkCount() const { return m_chunks.size(); }

private:
    QStringList tokenize(const QString& text) {
        QStringList words = text.toLower().split(QRegularExpression("\\W+"), QString::SkipEmptyParts);
        static QStringList stops = {"the","and","a","an","in","on","at","to","for","of","with","is","it","this","that","as","by","from","or","was","are","be","i","you","he","she","we","they","what","how","why","when","where"};
        QStringList res;
        for(const QString& w : words) {
            if (w.length() > 1 && !stops.contains(w)) res << w;
        }
        return res;
    }

    QVector<Chunk> m_chunks;
    QMap<QString, int> m_docFreq;
    QMap<QString, double> m_idf;
};

// ============================================================================
// MAIN UI WINDOW (No Q_OBJECT required, pure C++11 Lambdas)
// ============================================================================
class MainWindow : public QWidget {
public:
    MainWindow(QWidget* parent = nullptr) : QWidget(parent) {
        setWindowTitle("Extractive Thinking Agent - Qt 5.12");
        resize(900, 700);

        // Modern Dark IDE Theme
        setStyleSheet(
            "QWidget { background-color: #1e1e1e; color: #d4d4d4; font-family: 'Segoe UI', sans-serif; }"
            "QPushButton { background-color: #0e639c; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #1177bb; }"
            "QPushButton:disabled { background-color: #3e3e42; color: #888; }"
            "QLineEdit { background-color: #252526; border: 1px solid #3e3e42; padding: 10px; border-radius: 4px; color: white; font-size: 14px; }"
            "QTextEdit { background-color: #1e1e1e; border: 1px solid #3e3e42; color: #d4d4d4; font-family: 'Consolas', 'Courier New', monospace; }"
            "QLabel { background: transparent; }"
        );

        QVBoxLayout* mainLayout = new QVBoxLayout(this);
        mainLayout->setContentsMargins(15, 15, 15, 15);
        mainLayout->setSpacing(10);

        QHBoxLayout* header = new QHBoxLayout();
        QPushButton* btnTrain = new QPushButton("📚 Feed Encyclopedias / Books");
        QPushButton* btnClear = new QPushButton("🧹 Wipe Brain");
        m_lblStatus = new QLabel("Status: Awaiting Knowledge...");
        m_lblStatus->setStyleSheet("color: #888; font-size: 12px; margin-left: 10px;");

        header->addWidget(btnTrain);
        header->addWidget(btnClear);
        header->addSpacing(20);
        header->addWidget(m_lblStatus);
        header->addStretch();

        m_chatLog = new QTextEdit();
        m_chatLog->setReadOnly(true);
        m_chatLog->setHtml("<div style='color:#6a9955;'>[AGENT] I am an Extractive Thinking AI. Feed me .txt files (books, encyclopedias, docs), and I will read them to answer your questions accurately without hallucinating.</div>");

        QHBoxLayout* inputLayout = new QHBoxLayout();
        m_input = new QLineEdit();
        m_input->setPlaceholderText("Ask a question about your books...");
        QPushButton* btnAsk = new QPushButton("🧠 Think & Answer");
        btnAsk->setStyleSheet("background-color: #16825d;");

        inputLayout->addWidget(m_input);
        inputLayout->addWidget(btnAsk);

        mainLayout->addLayout(header);
        mainLayout->addWidget(m_chatLog);
        mainLayout->addLayout(inputLayout);

        // Logic connections using C++11 lambdas to avoid Q_OBJECT MOC issues
        QObject::connect(btnTrain, &QPushButton::clicked, [this]() {
            QStringList files = QFileDialog::getOpenFileNames(this, "Select Text Files", "", "Text Files (*.txt);;All Files (*)");
            if (files.isEmpty()) return;

            m_lblStatus->setText("Status: Ingesting...");
            QApplication::processEvents();

            for (const QString& f : files) {
                m_chatLog->append(QString("<span style='color:#569cd6;'>[LOAD] Reading %1...</span>").arg(QFileInfo(f).fileName()));
                QApplication::processEvents();

                m_brain.ingestFile(f);

                m_chatLog->append(QString("<span style='color:#4ec9b0;'>[DONE] Indexed %1.</span>").arg(QFileInfo(f).fileName()));
                QApplication::processEvents();
            }

            m_chatLog->append("<span style='color:#569cd6;'>[CALC] Computing semantic weights (IDF)...</span>");
            QApplication::processEvents();

            m_brain.computeIDF();

            m_lblStatus->setText(QString("Status: Ready (%1 chunks)").arg(m_brain.chunkCount()));
            m_chatLog->append("<hr><span style='color:#6a9955; font-weight:bold;'>[AGENT] I have digested the texts. Ask me anything!</span>");
            scrollToBottom();
        });

        QObject::connect(btnClear, &QPushButton::clicked, [this]() {
            m_brain.clear();
            m_chatLog->clear();
            m_lblStatus->setText("Status: Brain wiped.");
            m_chatLog->append("<span style='color:#ce9178;'>[SYSTEM] Memory cleared.</span>");
        });

        QObject::connect(btnAsk, &QPushButton::clicked, [this]() { processQuery(); });
        QObject::connect(m_input, &QLineEdit::returnPressed, [this]() { processQuery(); });
    }

private:
    void processQuery() {
        QString q = m_input->text().trimmed();
        if (q.isEmpty() || m_brain.chunkCount() == 0) {
            if (m_brain.chunkCount() == 0) QMessageBox::warning(this, "Empty Brain", "Please feed me some books first!");
            return;
        }

        m_chatLog->append(QString("<br><span style='color:#9cdcfe; font-weight:bold;'>&gt;&gt; USER:</span> %1").arg(q.toHtmlEscaped()));
        m_chatLog->append("<span style='color:#888; font-style:italic;'>[AGENT] Searching memory and extracting facts...</span>");
        QApplication::processEvents();
        scrollToBottom();

        QString answer = m_brain.thinkAndAnswer(q);

        QString formatted = answer.toHtmlEscaped().replace("\n", "<br>");
        m_chatLog->append(QString("<div style='background:#252526; border-left: 4px solid #16825d; padding: 10px; margin: 5px 0;'>"
                                  "<span style='color:#dcdcaa; font-weight:bold;'>🤖 Agent Synthesis:</span><br>"
                                  "<span style='color:#d4d4d4;'>%1</span></div><br>").arg(formatted));

        m_input->clear();
        scrollToBottom();
    }

    void scrollToBottom() {
        QScrollBar* sb = m_chatLog->verticalScrollBar();
        sb->setValue(sb->maximum());
    }

    QTextEdit* m_chatLog;
    QLineEdit* m_input;
    QLabel* m_lblStatus;
    AgentBrain m_brain;
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    MainWindow w;
    w.show();
    return app.exec();
}
