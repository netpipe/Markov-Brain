// ThinkingCodingAgent.pro equivalent - single file Qt 5.12 demo
// Build with: qmake && make OR add to Qt Creator
// Requires: Qt 5.12+

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
#include <QThread>
#include <QDebug>
#include <QFileInfo>
#include <QtMath>
#include <QScrollBar>
#include <QRandomGenerator>
#include <QFont>
#include <algorithm>

// ============================================================================
// KNOWLEDGE CHUNK STRUCTURE
// ============================================================================
struct Chunk {
    QString text;
    QString source;
    bool isCode;
    QMap<QString, int> termFreq; // Raw term frequencies
    double magnitude;            // Precomputed vector magnitude for cosine similarity
};

// ============================================================================
// THE "BRAIN" - Runs in a background thread to keep UI responsive
// ============================================================================
class CodingBrain : public QObject {
    Q_OBJECT
public:
    explicit CodingBrain(QObject* parent = nullptr) : QObject(parent) {
        // Common English stop words to ignore during text search
        m_stopWords = QStringList{"the", "and", "a", "an", "in", "on", "at", "to", "for", "of",
                                  "with", "is", "it", "this", "that", "as", "by", "from", "or",
                                  "was", "are", "be", "i", "you", "he", "she", "we", "they", "what"};
    }

public slots:
    void loadFiles(const QStringList& files) {
        emit thought("[SYSTEM] Initializing Knowledge Ingestion...");

        for (const QString& path : files) {
            QFile file(path);
            if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) continue;

            QString content = file.readAll();
            file.close();

            bool isCode = isCodeFile(path);
            emit thought(QString("[LOAD] Processing %1 (%2)...").arg(QFileInfo(path).fileName()).arg(isCode ? "Code" : "Text"));

            // Chunking strategy
            QStringList rawChunks;
            if (isCode) {
                // Chunk code by lines (approx 20 lines per chunk) to preserve context
                QStringList lines = content.split("\n");
                QString currentChunk;
                int lineCount = 0;
                for (const QString& line : lines) {
                    currentChunk += line + "\n";
                    lineCount++;
                    if (lineCount >= 20) {
                        rawChunks << currentChunk;
                        currentChunk = "";
                        lineCount = 0;
                    }
                }
                if (!currentChunk.isEmpty()) rawChunks << currentChunk;
            } else {
                // Chunk text by paragraphs
                rawChunks = content.split(QRegularExpression("\\n\\s*\\n"), QString::SkipEmptyParts);
            }

            for (const QString& raw : rawChunks) {
                if (raw.trimmed().isEmpty()) continue;
                Chunk c;
                c.text = raw;
                c.source = QFileInfo(path).fileName();
                c.isCode = isCode;

                QStringList tokens = tokenize(raw, isCode);
                for (const QString& t : tokens) {
                    c.termFreq[t]++;
                }

                // Train Code Markov Chain (Bigram model)
                if (isCode && tokens.size() > 1) {
                    m_codeStartTokens << tokens.first();
                    for (int i = 0; i < tokens.size() - 1; ++i) {
                        m_codeMarkov[tokens[i]].append(tokens[i+1]);
                    }
                }

                m_chunks.append(c);
            }
            emit fileLoaded(QFileInfo(path).fileName());
        }

        emit thought(QString("[SYSTEM] Computing TF-IDF vectors for %1 memory chunks...").arg(m_chunks.size()));
        computeIDF();

        // Precompute magnitudes for cosine similarity
        for (Chunk& c : m_chunks) {
            double mag = 0;
            for (const QString& term : c.termFreq.keys()) {
                double w = getWeight(c, term);
                mag += w * w;
            }
            c.magnitude = sqrt(mag);
        }

        emit thought("[SYSTEM] Knowledge Base ready. I am ready to think and code.");
        emit loadingComplete(m_chunks.size());
    }

    void askQuestion(const QString& query) {
        if (query.trimmed().isEmpty()) return;

        if (query.toLower() == "help") {
            emit response("I am a Thinking Coding Agent.\n"
                          "1. Click 'Feed Knowledge' and select .txt, .md, or code files (.cpp, .py, .js).\n"
                          "2. I will index them using TF-IDF and build a Markov brain for code syntax.\n"
                          "3. Ask me questions like 'How do I write a loop?' or 'Explain pointers'.\n"
                          "4. I will 'think', search my memory, and synthesize an answer!");
            return;
        }

        emit thought(QString("[THINK] Analyzing query: \"%1\"").arg(query));

        // 1. Tokenize query (treat as text to remove stop words)
        QStringList queryTokens = tokenize(query, false);
        if (queryTokens.isEmpty()) {
            emit response("I didn't catch any semantic keywords in your query.");
            return;
        }

        emit thought(QString("[THINK] Extracted %1 semantic tokens.").arg(queryTokens.size()));

        // 2. Compute query vector and magnitude
        QMap<QString, double> queryVec;
        double queryMag = 0;
        QMap<QString, int> qtf;
        for(const QString& t : queryTokens) qtf[t]++;

        for (auto it = qtf.begin(); it != qtf.end(); ++it) {
            double w = 1.0 + log10((double)it.value()) * m_idf.value(it.key(), 0.0);
            queryVec[it.key()] = w;
            queryMag += w * w;
        }
        queryMag = sqrt(queryMag);

        if (queryMag == 0) {
            emit response("I couldn't find those terms in my knowledge base.");
            return;
        }

        // 3. Search (Cosine Similarity)
        emit thought(QString("[SEARCH] Scanning %1 memory vectors...").arg(m_chunks.size()));

        struct SearchResult { int chunkIndex; double score; };
        QVector<SearchResult> results;

        for (int i = 0; i < m_chunks.size(); ++i) {
            const Chunk& c = m_chunks[i];
            double dotProduct = 0;
            for (auto it = queryVec.begin(); it != queryVec.end(); ++it) {
                double cw = getWeight(c, it.key());
                dotProduct += it.value() * cw;
            }

            double score = 0;
            if (c.magnitude > 0 && queryMag > 0) {
                score = dotProduct / (c.magnitude * queryMag);
            }
            if (score > 0.01) { // Relevance threshold
                results.append({i, score});
            }
        }

        // Sort descending by relevance
        std::sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& b) {
            return a.score > b.score;
        });

        if (results.isEmpty()) {
            emit thought("[THINK] No relevant memories found.");
            emit response("I don't know anything about that yet. Feed me more encyclopedias or code!");
            return;
        }

        int topN = qMin(3, results.size());
        emit thought(QString("[RETRIEVE] Found %1 relevant context(s). Synthesizing...").arg(topN));

        // 4. Synthesize Response (RAG - Retrieval Augmented Generation)
        QString finalResponse;
        bool hasCode = false;

        for (int i = 0; i < topN; ++i) {
            const Chunk& c = m_chunks[results[i].chunkIndex];
            if (c.isCode) hasCode = true;
            finalResponse += QString(">> From %1:\n%2\n\n").arg(c.source, c.text.trimmed());
        }

        // 5. If it's a code question, attempt to generate a snippet using Markov Chain
        bool isCodeQuery = query.contains("code", Qt::CaseInsensitive) ||
                           query.contains("write", Qt::CaseInsensitive) ||
                           query.contains("function", Qt::CaseInsensitive) ||
                           query.contains("script", Qt::CaseInsensitive) ||
                           hasCode;

        if (isCodeQuery && !m_codeMarkov.isEmpty()) {
            emit thought("[GENERATE] Attempting to synthesize code pattern via Markov Brain...");
            QString generatedCode = generateCode(queryTokens);
            if (!generatedCode.isEmpty()) {
                finalResponse += ">> Synthesized Code Snippet:\n```cpp\n" + generatedCode + "\n```\n";
            }
        }

        emit response(finalResponse);
    }

    void clearBrain() {
        m_chunks.clear();
        m_idf.clear();
        m_codeMarkov.clear();
        m_codeStartTokens.clear();
        emit thought("[SYSTEM] Memory wiped. Awaiting new data...");
        emit loadingComplete(0);
    }

signals:
    void thought(const QString& msg);
    void response(const QString& msg);
    void fileLoaded(const QString& fileName);
    void loadingComplete(int totalChunks);

private:
    QVector<Chunk> m_chunks;
    QMap<QString, double> m_idf;
    int m_totalDocs;

    // Code Markov Chain (Bigram)
    QMap<QString, QStringList> m_codeMarkov;
    QStringList m_codeStartTokens;

    QStringList m_stopWords;

    bool isCodeFile(const QString& path) {
        return path.endsWith(".cpp", Qt::CaseInsensitive) ||
               path.endsWith(".c", Qt::CaseInsensitive) ||
               path.endsWith(".h", Qt::CaseInsensitive) ||
               path.endsWith(".py", Qt::CaseInsensitive) ||
               path.endsWith(".js", Qt::CaseInsensitive) ||
               path.endsWith(".java", Qt::CaseInsensitive) ||
               path.endsWith(".cs", Qt::CaseInsensitive);
    }

    QStringList tokenize(const QString& text, bool isCode) {
        QStringList tokens;
        if (isCode) {
            // Tokenize code by separating symbols and words
            QRegularExpression re("(\\w+|[{}();,\\[\\]\\+\\-\\*/=<>!&\\|])");
            QRegularExpressionMatchIterator i = re.globalMatch(text);
            while (i.hasNext()) {
                tokens << i.next().captured(1);
            }
        } else {
            // Tokenize text, lowercase, remove stop words
            QStringList words = text.split(QRegularExpression("\\W+"), QString::SkipEmptyParts);
            for(QString& w : words) {
                w = w.toLower();
                if (!m_stopWords.contains(w) && w.length() > 1) {
                    tokens << w;
                }
            }
        }
        return tokens;
    }

    void computeIDF() {
        m_idf.clear();
        QMap<QString, int> docFreq;
        for (const Chunk& c : m_chunks) {
            for (const QString& term : c.termFreq.keys()) {
                docFreq[term]++;
            }
        }
        m_totalDocs = m_chunks.size();
        for (auto it = docFreq.begin(); it != docFreq.end(); ++it) {
            m_idf[it.key()] = log10((double)m_totalDocs / (double)it.value());
        }
    }

    double getWeight(const Chunk& c, const QString& term) {
        int tf = c.termFreq.value(term, 0);
        if (tf == 0) return 0.0;
        double subTf = 1.0 + log10((double)tf); // Sublinear TF
        double idf = m_idf.value(term, 0.0);
        return subTf * idf;
    }

    QString generateCode(const QStringList& seedWords) {
        if (m_codeMarkov.isEmpty()) return "";

        // Find a starting token that matches a seed word
        QString startToken = "";
        for (const QString& w : seedWords) {
            for (const QString& key : m_codeMarkov.keys()) {
                if (key.compare(w, Qt::CaseInsensitive) == 0) {
                    startToken = key;
                    break;
                }
            }
            if (!startToken.isEmpty()) break;
        }

        if (startToken.isEmpty()) {
            QStringList keys = m_codeMarkov.keys();
            startToken = keys.at(QRandomGenerator::global()->bounded(keys.size()));
        }

        QStringList result;
        result << startToken;
        QString current = startToken;

        for (int i = 0; i < 40; ++i) {
            if (!m_codeMarkov.contains(current)) break;
            QStringList nexts = m_codeMarkov[current];
            if (nexts.isEmpty()) break;

            QString next = nexts.at(QRandomGenerator::global()->bounded(nexts.size()));
            result << next;
            current = next;

            if (next == "}" && result.size() > 5) break;
            if (next == ";" && result.size() > 15) break;
        }

        // Simple formatting
        QString code = "";
        for (const QString& token : result) {
            if (token == "{" || token == "}") {
                code += " " + token + "\n";
            } else if (token == ";") {
                code += token + "\n";
            } else {
                code += token + " ";
            }
        }
        return code.trimmed();
    }
};

// ============================================================================
// MAIN UI WINDOW
// ============================================================================
class MainWindow : public QWidget {
    Q_OBJECT
public:
    MainWindow(QWidget* parent = nullptr) : QWidget(parent) {
        setWindowTitle("Thinking Coding Agent - Qt 5.12 RAG/Markov Demo");
        resize(950, 750);

        // Dark IDE Theme
        setStyleSheet(
            "QPushButton { padding: 8px 15px; background: #2c3e50; color: white; border: none; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #34495e; }"
            "QLineEdit { padding: 10px; border: 1px solid #7f8c8d; border-radius: 4px; background: #ecf0f1; }"
            "QTextEdit { border: 1px solid #7f8c8d; background: #1e1e1e; color: #dcdcdc; }"
            "QLabel { color: #ecf0f1; font-weight: bold; }"
            "QWidget { background: #2b2b2b; }"
        );

        QVBoxLayout* mainLayout = new QVBoxLayout(this);

        // Top Bar
        QHBoxLayout* topBar = new QHBoxLayout();
        QPushButton* btnLoad = new QPushButton("📂 Feed Knowledge");
        QPushButton* btnClear = new QPushButton("🗑️ Clear Memory");
        m_lblStatus = new QLabel("Status: Brain Empty. Feed me data!");
        topBar->addWidget(btnLoad);
        topBar->addWidget(btnClear);
        topBar->addSpacing(20);
        topBar->addWidget(m_lblStatus);
        topBar->addStretch();

        // Chat Log
        m_chatLog = new QTextEdit();
        m_chatLog->setReadOnly(true);
        m_chatLog->setFont(QFont("Consolas", 10));

        // Input Bar
        QHBoxLayout* inputBar = new QHBoxLayout();
        m_input = new QLineEdit();
        m_input->setPlaceholderText("Ask me to write code, explain concepts, or search the encyclopedia...");
        QPushButton* btnSend = new QPushButton("🧠 Think & Answer");
        btnSend->setStyleSheet("background: #27ae60;");
        inputBar->addWidget(m_input);
        inputBar->addWidget(btnSend);

        mainLayout->addLayout(topBar);
        mainLayout->addWidget(m_chatLog);
        mainLayout->addLayout(inputBar);

        // Threading setup for the Brain
        m_brain = new CodingBrain();
        m_thread = new QThread(this);
        m_brain->moveToThread(m_thread);

        connect(m_thread, &QThread::finished, m_brain, &QObject::deleteLater);
        connect(this, &MainWindow::requestLoad, m_brain, &CodingBrain::loadFiles);
        connect(this, &MainWindow::requestAsk, m_brain, &CodingBrain::askQuestion);
        connect(this, &MainWindow::requestClear, m_brain, &CodingBrain::clearBrain);

        connect(m_brain, &CodingBrain::thought, this, &MainWindow::appendThought);
        connect(m_brain, &CodingBrain::response, this, &MainWindow::appendResponse);
        connect(m_brain, &CodingBrain::fileLoaded, this, &MainWindow::onFileLoaded);
        connect(m_brain, &CodingBrain::loadingComplete, this, [this](int chunks){
            m_lblStatus->setText(QString("Status: Active (%1 memory chunks)").arg(chunks));
        });

        m_thread->start();

        // UI Signals
        connect(btnLoad, &QPushButton::clicked, this, &MainWindow::selectFiles);
        connect(btnClear, &QPushButton::clicked, this, [this]() {
            m_chatLog->append("<hr>");
            emit requestClear();
        });
        connect(btnSend, &QPushButton::clicked, this, &MainWindow::sendQuery);
        connect(m_input, &QLineEdit::returnPressed, this, &MainWindow::sendQuery);

        appendThought("[SYSTEM] Agent initialized. Awaiting knowledge ingestion...");
        appendThought("[TIP] Type 'help' to see what I can do.");
    }

    ~MainWindow() {
        m_thread->quit();
        m_thread->wait();
    }

signals:
    void requestLoad(const QStringList& files);
    void requestAsk(const QString& query);
    void requestClear();

private slots:
    void selectFiles() {
        QStringList files = QFileDialog::getOpenFileNames(this, "Select Knowledge Sources (Books, Encyclopedias, Code)", "",
            "All Supported (*.txt *.md *.cpp *.c *.h *.py *.js *.java);;Text (*.txt *.md);;Code (*.cpp *.c *.h *.py *.js *.java)");
        if (!files.isEmpty()) {
            m_lblStatus->setText("Status: Ingesting data...");
            m_chatLog->append("<hr>");
            emit requestLoad(files);
        }
    }

    void sendQuery() {
        QString q = m_input->text().trimmed();
        if (!q.isEmpty()) {
            m_chatLog->append(QString("<br><span style='color:#3498db; font-weight:bold;'>&gt;&gt; USER:</span> %1").arg(q.toHtmlEscaped()));
            m_input->clear();
            emit requestAsk(q);
        }
    }

    void appendThought(const QString& msg) {
        m_chatLog->append(QString("<span style='color:#95a5a6; font-family: Consolas, monospace; font-size: 11px;'>%1</span>").arg(msg.toHtmlEscaped()));
        scrollToBottom();
    }

    void appendResponse(const QString& msg) {
        QString escaped = msg.toHtmlEscaped();
        m_chatLog->append(QString("<br><span style='color:#2ecc71; font-weight:bold; font-size:14px;'>&gt;&gt; AGENT SYNTHESIS</span><br>"
                                  "<div style='background:#252526; border-left: 4px solid #2ecc71; padding: 12px; margin: 5px 0; white-space: pre-wrap; font-family: Consolas, monospace; color: #dcdcdc;'>%1</div><br>")
                                  .arg(escaped));
        scrollToBottom();
    }

    void onFileLoaded(const QString& fileName) {
        appendThought(QString("[SYSTEM] Ingested: %1").arg(fileName));
    }

    void scrollToBottom() {
        QScrollBar* sb = m_chatLog->verticalScrollBar();
        sb->setValue(sb->maximum());
    }

private:
    QTextEdit* m_chatLog;
    QLineEdit* m_input;
    QLabel* m_lblStatus;

    CodingBrain* m_brain;
    QThread* m_thread;
};

// ============================================================================
// ENTRY POINT
// ============================================================================
int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    MainWindow w;
    w.show();
    return app.exec();
}

#include "main.moc"
