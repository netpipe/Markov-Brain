// main.cpp - Vision-to-Speech Brain Application (Qt 5.12)
// Fixed ANN integration with proper nested class definitions
 // untested use at own risk still

#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QTabWidget>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QTextEdit>
#include <QPlainTextEdit>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QProgressBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QInputDialog>
#include <QImage>
#include <QPixmap>
#include <QScrollArea>
#include <QGroupBox>
#include <QComboBox>
#include <QCheckBox>
#include <QTimer>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QSqlError>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QDataStream>
#include <QThread>
#include <QMutex>
#include <QDateTime>
#include <QListWidget>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QSplitter>
#include <QStatusBar>
#include <QMenuBar>
#include <QAction>
#include <QPainter>
#include <QBuffer>
#include <QProgressDialog>
#include <QtMath>
#include <QTextStream>
#include <vector>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <map>
#include <memory>
#include <functional>
#include <numeric>

using namespace std;

// ============================================================
// ANN ENGINE - Fixed nested class definitions
// ============================================================

static const float g_learningRate = 0.05f;

class Ann
{
public:
    class Link;
    class BackLink;
    class INode;

    class ITrainer
    {
    public:
        virtual ~ITrainer() {}
        virtual void train(std::vector<float>& inputs,
                           std::vector<float>& outputs, Ann* ann) = 0;
    };

    Ann(size_t numberOfWorkLayers = 2, size_t numberOfWorkers = 8,
        size_t numberOfInputs = 3, size_t numberOfOutputs = 3);
    ~Ann();

    void setTrainer(Ann::ITrainer* trainerIn) { trainer = trainerIn; }
    void backPropagate(size_t iterations);
    void randomizeInputs();
    void setInput(const std::vector<float>& input);
    std::vector<float> getOutput();
    std::vector<float> getInput();
    void saveNeuralMap(const char* filename);
    bool loadNeuralMap(const char* filename);

    std::vector<float> exportWeights();
    bool importWeights(const std::vector<float>& weights);
    size_t getInputSize() const { return inputLayerSize; }
    size_t getOutputSize() const { return outputLayerSize; }

private:
    Ann::ITrainer* trainer;
    std::vector<Ann::INode*> inputLayer;
    std::vector<std::vector<Ann::INode*> > workLayers;
    std::vector<Ann::INode*> outputLayer;
    size_t inputLayerSize;
    size_t workLayersSize;
    size_t workLayerSize;
    size_t outputLayerSize;
    std::vector<float> inputValues;
    std::vector<float> outputValues;
};

// --- INode (base node) ---
class Ann::INode
{
public:
    virtual ~INode() {}
    virtual float getOutput() = 0;
    virtual void calculateDelta() {}
    virtual void updateFreeValues() {}
    std::vector<Ann::Link>& getLinks() { return links; }
    float delta;
    float bias;
    std::vector<Ann::BackLink> backLinks;
protected:
    std::vector<Ann::Link> links;
};

// --- Link ---
class Ann::Link
{
public:
    Link(Ann::INode* nodeLink) : pointer(nodeLink)
    {
        weight = (rand() % 400) * 0.01f - 2.0f;
    }
    float weight;
    Ann::INode* pointer;
};

// --- BackLink ---
class Ann::BackLink
{
public:
    BackLink(Ann::INode* nodeLink, float* weightIn)
        : pointer(nodeLink), weight(weightIn) {}
    float* weight;
    Ann::INode* pointer;
};

// --- BasicNode ---
class BasicNode : public Ann::INode
{
public:
    BasicNode(std::vector<Ann::INode*>& inputs)
    {
        bias = (rand() % 100) * 0.01f;
        size_t inputsSize = inputs.size();
        links.reserve(inputsSize);
        for (size_t i = 0; i < inputsSize; ++i)
        {
            links.push_back(Ann::Link(inputs[i]));
            inputs[i]->backLinks.push_back(Ann::BackLink(this, &links[i].weight));
        }
    }

    virtual float getOutput() override
    {
        float netValue = bias;
        size_t linksSize = links.size();
        for (size_t i = 0; i < linksSize; ++i)
            netValue += links[i].pointer->getOutput() * links[i].weight;
        netValue = 1.0f / (1.0f + exp(-netValue));
        return netValue;
    }

    virtual void calculateDelta() override
    {
        float errorFactor = 0.0f;
        size_t backLinksSize = backLinks.size();
        for (size_t i = 0; i < backLinksSize; ++i)
            errorFactor += (*backLinks[i].weight) * backLinks[i].pointer->delta;
        float currentOutput = getOutput();
        delta = currentOutput * (1.0f - currentOutput) * errorFactor;
    }

    virtual void updateFreeValues() override
    {
        bias = bias + g_learningRate * delta;
        size_t currentLinksSize = links.size();
        for (size_t g = 0; g < currentLinksSize; ++g)
            links[g].weight = links[g].weight + g_learningRate
                * links[g].pointer->getOutput() * delta;
    }
};

// --- InputNode ---
class InputNode : public Ann::INode
{
public:
    InputNode(float inputValue) : value(inputValue) {}
    virtual float getOutput() override { return value; }
    float value;
};

// --- Ann implementation ---
Ann::Ann(size_t numberOfWorkLayers, size_t numberOfWorkers,
         size_t numberOfInputs, size_t numberOfOutputs)
    : trainer(0)
{
    srand((unsigned)clock());
    for (size_t i = 0; i < numberOfInputs; ++i)
        inputLayer.push_back(new InputNode(0));
    inputLayerSize = inputLayer.size();

    workLayers.push_back(inputLayer);
    for (size_t g = 0; g < numberOfWorkLayers; ++g)
    {
        std::vector<Ann::INode*> workLayer;
        for (size_t i = 0; i < numberOfWorkers; ++i)
            workLayer.push_back(new BasicNode(workLayers[g]));
        workLayers.push_back(workLayer);
    }
    workLayerSize = numberOfWorkers;
    workLayersSize = workLayers.size();
    for (size_t i = 0; i < numberOfOutputs; ++i)
        outputLayer.push_back(new BasicNode(workLayers[workLayersSize - 1]));
    outputLayerSize = outputLayer.size();
}

Ann::~Ann()
{
    for (size_t i = 0; i < inputLayerSize; ++i)
        delete inputLayer[i];
    for (size_t g = 1; g < workLayersSize; ++g)
        for (size_t i = 0; i < workLayerSize; ++i)
            delete workLayers[g][i];
    for (size_t i = 0; i < outputLayerSize; ++i)
        delete outputLayer[i];
}

void Ann::randomizeInputs()
{
    for (size_t i = 0; i < inputLayerSize; ++i)
        ((InputNode*)inputLayer[i])->value = inputValues[i] = (rand() % 100) * 0.01f;
}

void Ann::backPropagate(size_t iterations)
{
    for (size_t l = 0; l < iterations; ++l)
    {
        inputValues.resize(inputLayerSize);
        outputValues.resize(outputLayerSize);
        if (trainer)
            trainer->train(inputValues, outputValues, this);
        for (size_t i = 0; i < inputLayerSize; ++i)
            ((InputNode*)inputLayer[i])->value = inputValues[i];
        for (size_t i = 0; i < outputLayerSize; ++i)
        {
            float outputValue = outputLayer[i]->getOutput();
            float outputEFactor = outputValues[i] - outputValue;
            outputLayer[i]->delta = outputValue * (1.0f - outputValue) * outputEFactor;
        }
        for (int i = (int)workLayersSize - 1; i >= 1; --i)
        {
            for (size_t g = 0; g < workLayerSize; ++g)
            {
                workLayers[i][g]->calculateDelta();
                workLayers[i][g]->updateFreeValues();
            }
        }
        for (size_t i = 0; i < outputLayerSize; ++i)
            outputLayer[i]->updateFreeValues();
    }
}

void Ann::setInput(const std::vector<float>& input)
{
    size_t inputSize = input.size();
    for (size_t i = 0; i < inputLayerSize && i < inputSize; ++i)
        ((InputNode*)inputLayer[i])->value = input[i];
}

std::vector<float> Ann::getOutput()
{
    std::vector<float> output;
    for (size_t i = 0; i < outputLayerSize; ++i)
        output.push_back(outputLayer[i]->getOutput());
    return output;
}

std::vector<float> Ann::getInput()
{
    std::vector<float> input;
    for (size_t i = 0; i < inputLayerSize; ++i)
        input.push_back(inputLayer[i]->getOutput());
    return input;
}

void Ann::saveNeuralMap(const char* filename)
{
    FILE* file = fopen(filename, "wb");
    if (!file) return;
    fwrite(&workLayersSize, sizeof(size_t), 1, file);
    fwrite(&workLayerSize, sizeof(size_t), 1, file);
    fwrite(&inputLayerSize, sizeof(size_t), 1, file);
    fwrite(&outputLayerSize, sizeof(size_t), 1, file);
    for (size_t g = 1; g < workLayersSize; ++g)
    {
        for (size_t i = 0; i < workLayerSize; ++i)
        {
            fwrite(&workLayers[g][i]->bias, sizeof(float), 1, file);
            size_t linksSize = workLayers[g][i]->getLinks().size();
            for (size_t j = 0; j < linksSize; ++j)
                fwrite(&workLayers[g][i]->getLinks()[j].weight, sizeof(float), 1, file);
        }
    }
    for (size_t i = 0; i < outputLayerSize; ++i)
    {
        fwrite(&outputLayer[i]->bias, sizeof(float), 1, file);
        size_t linksSize = outputLayer[i]->getLinks().size();
        for (size_t j = 0; j < linksSize; ++j)
            fwrite(&outputLayer[i]->getLinks()[j].weight, sizeof(float), 1, file);
    }
    fclose(file);
}

bool Ann::loadNeuralMap(const char* filename)
{
    FILE* file = fopen(filename, "rb");
    if (!file) return false;
    size_t fWorkLayersSize = 0, fWorkLayerSize = 0;
    size_t fInputLayersSize = 0, fOutputLayersSize = 0;
    fread(&fWorkLayersSize, sizeof(size_t), 1, file);
    fread(&fWorkLayerSize, sizeof(size_t), 1, file);
    fread(&fInputLayersSize, sizeof(size_t), 1, file);
    fread(&fOutputLayersSize, sizeof(size_t), 1, file);
    if (fWorkLayersSize != workLayersSize || fWorkLayerSize != workLayerSize ||
        fInputLayersSize != inputLayerSize || fOutputLayersSize != outputLayerSize)
    {
        fclose(file);
        return false;
    }
    for (size_t g = 1; g < workLayersSize; ++g)
    {
        for (size_t i = 0; i < workLayerSize; ++i)
        {
            fread(&workLayers[g][i]->bias, sizeof(float), 1, file);
            size_t linksSize = workLayers[g][i]->getLinks().size();
            for (size_t j = 0; j < linksSize; ++j)
                fread(&workLayers[g][i]->getLinks()[j].weight, sizeof(float), 1, file);
        }
    }
    for (size_t i = 0; i < outputLayerSize; ++i)
    {
        fread(&outputLayer[i]->bias, sizeof(float), 1, file);
        size_t linksSize = outputLayer[i]->getLinks().size();
        for (size_t j = 0; j < linksSize; ++j)
            fread(&outputLayer[i]->getLinks()[j].weight, sizeof(float), 1, file);
    }
    fclose(file);
    return true;
}

std::vector<float> Ann::exportWeights()
{
    std::vector<float> weights;
    for (size_t g = 1; g < workLayersSize; ++g)
        for (size_t i = 0; i < workLayerSize; ++i)
        {
            weights.push_back(workLayers[g][i]->bias);
            size_t ls = workLayers[g][i]->getLinks().size();
            for (size_t j = 0; j < ls; ++j)
                weights.push_back(workLayers[g][i]->getLinks()[j].weight);
        }
    for (size_t i = 0; i < outputLayerSize; ++i)
    {
        weights.push_back(outputLayer[i]->bias);
        size_t ls = outputLayer[i]->getLinks().size();
        for (size_t j = 0; j < ls; ++j)
            weights.push_back(outputLayer[i]->getLinks()[j].weight);
    }
    return weights;
}

bool Ann::importWeights(const std::vector<float>& weights)
{
    size_t idx = 0;
    for (size_t g = 1; g < workLayersSize && idx < weights.size(); ++g)
        for (size_t i = 0; i < workLayerSize && idx < weights.size(); ++i)
        {
            workLayers[g][i]->bias = weights[idx++];
            size_t ls = workLayers[g][i]->getLinks().size();
            for (size_t j = 0; j < ls && idx < weights.size(); ++j)
                workLayers[g][i]->getLinks()[j].weight = weights[idx++];
        }
    for (size_t i = 0; i < outputLayerSize && idx < weights.size(); ++i)
    {
        outputLayer[i]->bias = weights[idx++];
        size_t ls = outputLayer[i]->getLinks().size();
        for (size_t j = 0; j < ls && idx < weights.size(); ++j)
            outputLayer[i]->getLinks()[j].weight = weights[idx++];
    }
    return true;
}

// ============================================================
// CONSTANTS
// ============================================================

static const int IMAGE_SIZE = 16;
static const int INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE; // 256
static const int MAX_VOCAB = 128;
static const int OUTPUT_SIZE = MAX_VOCAB;
static const int HIDDEN_LAYERS = 2;
static const int HIDDEN_NEURONS = 64;

// ============================================================
// IMAGE PROCESSING
// ============================================================

class ImageProcessor
{
public:
    static std::vector<float> imageToVector(const QImage& img)
    {
        QImage scaled = img.scaled(IMAGE_SIZE, IMAGE_SIZE,
                                   Qt::IgnoreAspectRatio, Qt::SmoothTransformation)
                           .convertToFormat(QImage::Format_Grayscale8);
        std::vector<float> pixels(INPUT_SIZE);
        for (int y = 0; y < IMAGE_SIZE; ++y)
            for (int x = 0; x < IMAGE_SIZE; ++x)
                pixels[y * IMAGE_SIZE + x] = qGray(scaled.pixel(x, y)) / 255.0f;
        return pixels;
    }
};

// ============================================================
// VOCABULARY
// ============================================================

class Vocabulary
{
public:
    int addWord(const QString& word)
    {
        QString w = word.toLower().trimmed();
        if (w.isEmpty()) return -1;
        if (wordToIndex.contains(w)) return wordToIndex[w];
        if (indexToWord.size() >= MAX_VOCAB) return -1;
        int idx = (int)indexToWord.size();
        wordToIndex[w] = idx;
        indexToWord.append(w);
        return idx;
    }

    int getWordIndex(const QString& word) const
    {
        QString w = word.toLower().trimmed();
        if (wordToIndex.contains(w)) return wordToIndex[w];
        return -1;
    }

    QString getWord(int index) const
    {
        if (index >= 0 && index < indexToWord.size()) return indexToWord[index];
        return "?";
    }

    int size() const { return indexToWord.size(); }

    std::vector<float> encodeWord(const QString& word) const
    {
        std::vector<float> out(OUTPUT_SIZE, 0.0f);
        int idx = getWordIndex(word);
        if (idx >= 0 && idx < OUTPUT_SIZE) out[idx] = 1.0f;
        return out;
    }

    QString decodeOutput(const std::vector<float>& output, float* confidence = nullptr) const
    {
        if (output.empty()) return "?";
        int bestIdx = 0;
        float bestVal = output[0];
        int limit = qMin((int)output.size(), indexToWord.size());
        for (int i = 1; i < limit; ++i)
        {
            if (output[i] > bestVal) { bestVal = output[i]; bestIdx = i; }
        }
        if (confidence) *confidence = bestVal;
        return getWord(bestIdx);
    }

    void clear() { wordToIndex.clear(); indexToWord.clear(); }
    const QVector<QString>& getWordList() const { return indexToWord; }

private:
    QMap<QString, int> wordToIndex;
    QVector<QString> indexToWord;
};

// ============================================================
// DATABASE MANAGER
// ============================================================

class DatabaseManager
{
public:
    DatabaseManager() {}
    ~DatabaseManager() { if (db.isOpen()) db.close(); }

    bool open(const QString& path)
    {
        db = QSqlDatabase::addDatabase("QSQLITE");
        db.setDatabaseName(path);
        if (!db.open()) return false;
        createTables();
        return true;
    }

    void close() { if (db.isOpen()) db.close(); }
    QSqlDatabase& getDb() { return db; }

private:
    void createTables()
    {
        QSqlQuery q(db);
        q.exec("CREATE TABLE IF NOT EXISTS images ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT,"
               "filepath TEXT NOT NULL,"
               "description TEXT,"
               "created_at DATETIME DEFAULT CURRENT_TIMESTAMP)");
        q.exec("CREATE TABLE IF NOT EXISTS words ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT,"
               "word TEXT UNIQUE NOT NULL)");
        q.exec("CREATE TABLE IF NOT EXISTS image_word_pairs ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT,"
               "image_id INTEGER, word_id INTEGER,"
               "FOREIGN KEY(image_id) REFERENCES images(id),"
               "FOREIGN KEY(word_id) REFERENCES words(id))");
        q.exec("CREATE TABLE IF NOT EXISTS training_sessions ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT,"
               "session_name TEXT, iterations INTEGER,"
               "samples INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)");
        q.exec("CREATE TABLE IF NOT EXISTS neural_maps ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT,"
               "name TEXT, filepath TEXT,"
               "created_at DATETIME DEFAULT CURRENT_TIMESTAMP)");
        q.exec("CREATE TABLE IF NOT EXISTS brain_fusions ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT,"
               "name TEXT, source_maps TEXT, output_map TEXT,"
               "method TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)");
        q.exec("CREATE TABLE IF NOT EXISTS conversation_log ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT,"
               "input_image TEXT, response TEXT, confidence REAL,"
               "created_at DATETIME DEFAULT CURRENT_TIMESTAMP)");
    }

    QSqlDatabase db;
};

// ============================================================
// BRAIN FUSION ENGINE
// ============================================================

class BrainFusionEngine
{
public:
    enum FusionMethod { AVERAGE, WEIGHTED_AVERAGE, MAX_ACTIVATION, MEDIAN_MERGE };

    static bool fuseNeuralMaps(const QStringList& mapFiles, const QString& outputFile,
                               FusionMethod method = AVERAGE)
    {
        if (mapFiles.size() < 2) return false;

        // Read architecture from first file
        FILE* f = fopen(mapFiles[0].toStdString().c_str(), "rb");
        if (!f) return false;
        size_t wls, wlsz, ils, ols;
        fread(&wls, sizeof(size_t), 1, f);
        fread(&wlsz, sizeof(size_t), 1, f);
        fread(&ils, sizeof(size_t), 1, f);
        fread(&ols, sizeof(size_t), 1, f);
        fclose(f);

        Ann tempNet(wls - 1, wlsz, ils, ols);
        std::vector<std::vector<float>> allWeights;

        for (int i = 0; i < mapFiles.size(); ++i)
        {
            if (tempNet.loadNeuralMap(mapFiles[i].toStdString().c_str()))
                allWeights.push_back(tempNet.exportWeights());
        }

        if (allWeights.size() < 2) return false;

        std::vector<float> fused = fuseWeights(allWeights, method);
        tempNet.importWeights(fused);
        tempNet.saveNeuralMap(outputFile.toStdString().c_str());
        return true;
    }

private:
    static std::vector<float> fuseWeights(
        const std::vector<std::vector<float>>& all, FusionMethod method)
    {
        if (all.empty()) return {};
        size_t sz = all[0].size();
        std::vector<float> result(sz, 0.0f);

        switch (method)
        {
        case AVERAGE:
            for (size_t i = 0; i < sz; ++i)
            {
                float sum = 0.0f;
                for (const auto& w : all) sum += (i < w.size()) ? w[i] : 0.0f;
                result[i] = sum / (float)all.size();
            }
            break;
        case WEIGHTED_AVERAGE:
            // Weight later networks more heavily
            {
                float totalW = 0.0f;
                for (size_t n = 0; n < all.size(); ++n) totalW += (float)(n + 1);
                for (size_t i = 0; i < sz; ++i)
                {
                    float sum = 0.0f;
                    for (size_t n = 0; n < all.size(); ++n)
                        sum += ((i < all[n].size()) ? all[n][i] : 0.0f) * (float)(n + 1);
                    result[i] = sum / totalW;
                }
            }
            break;
        case MAX_ACTIVATION:
            for (size_t i = 0; i < sz; ++i)
            {
                float maxAbs = 0.0f;
                for (const auto& w : all)
                    if (i < w.size() && fabs(w[i]) > maxAbs)
                    { maxAbs = fabs(w[i]); result[i] = w[i]; }
            }
            break;
        case MEDIAN_MERGE:
            for (size_t i = 0; i < sz; ++i)
            {
                std::vector<float> vals;
                for (const auto& w : all)
                    if (i < w.size()) vals.push_back(w[i]);
                std::sort(vals.begin(), vals.end());
                result[i] = vals.empty() ? 0.0f : vals[vals.size() / 2];
            }
            break;
        }
        return result;
    }
};

// ============================================================
// TRAINING DATA
// ============================================================

struct TrainingSample
{
    std::vector<float> input;
    std::vector<float> output;
    QString word;
    QString imagePath;
};

class TrainingDataProvider : public Ann::ITrainer
{
public:
    TrainingDataProvider() : currentIndex(0) {}

    void setTrainingData(const std::vector<TrainingSample>& data)
    {
        samples = data;
        currentIndex = 0;
    }

    void addSample(const TrainingSample& s) { samples.push_back(s); }
    size_t count() const { return samples.size(); }
    void resetIndex() { currentIndex = 0; }

    virtual void train(std::vector<float>& inputs,
                       std::vector<float>& outputs, Ann*) override
    {
        if (samples.empty()) return;
        if (currentIndex >= samples.size()) currentIndex = 0;
        inputs = samples[currentIndex].input;
        outputs = samples[currentIndex].output;
        currentIndex++;
    }

private:
    std::vector<TrainingSample> samples;
    size_t currentIndex;
};

// ============================================================
// TRAINING WORKER THREAD
// ============================================================

class TrainingWorker : public QThread
{
    Q_OBJECT
public:
    explicit TrainingWorker(QObject* parent = nullptr)
        : QThread(parent), ann(nullptr), iterations(1000), shouldStop(false) {}

    void setNetwork(Ann* net) { ann = net; }
    void setIterations(int it) { iterations = it; }
    void requestStop() { shouldStop = true; }

signals:
    void progressUpdated(int current, int total);
    void finished();
    void errorOccurred(const QString& msg);

protected:
    void run() override
    {
        if (!ann) { emit errorOccurred("No network"); return; }
        int batch = 10;
        int total = iterations;
        for (int done = 0; done < total && !shouldStop; done += batch)
        {
            int count = qMin(batch, total - done);
            ann->backPropagate(count);
            emit progressUpdated(done + count, total);
            msleep(1);
        }
        if (!shouldStop) emit finished();
    }

private:
    Ann* ann;
    int iterations;
    volatile bool shouldStop;
};

// ============================================================
// MAIN WINDOW
// ============================================================

class VisionBrainApp : public QMainWindow
{
    Q_OBJECT

public:
    VisionBrainApp(QWidget* parent = nullptr)
        : QMainWindow(parent), ann(nullptr), worker(nullptr)
    {
        setWindowTitle("Vision-to-Speech Brain - Neural Learning Studio");
        resize(1100, 750);

        ann = new Ann(HIDDEN_LAYERS, HIDDEN_NEURONS, INPUT_SIZE, OUTPUT_SIZE);

        setupUi();
        setupDatabase();
        setupMenu();
        statusBar()->showMessage("Ready. Network: 256->64->64->128");
    }

    ~VisionBrainApp()
    {
        if (worker) { worker->requestStop(); worker->wait(3000); delete worker; }
        delete ann;
        dbMgr.close();
    }

private slots:
    void onLoadImages()
    {
        QStringList files = QFileDialog::getOpenFileNames(this, "Select Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.xpm)");
        if (files.isEmpty()) return;

        for (const QString& file : files)
        {
            QImage img(file);
            if (img.isNull()) continue;

            bool ok;
            QString word = QInputDialog::getText(this, "Label Image",
                "What is this?\n" + QFileInfo(file).fileName(),
                QLineEdit::Normal, "", &ok);
            if (!ok || word.isEmpty()) continue;

            addTrainingPair(file, img, word.toLower().trimmed());
        }
        updateSampleList();
    }

    void onBatchLabel()
    {
        QStringList files = QFileDialog::getOpenFileNames(this, "Select Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.xpm)");
        if (files.isEmpty()) return;

        bool ok;
        QString word = QInputDialog::getText(this, "Batch Label",
            "Label for ALL " + QString::number(files.size()) + " images:",
            QLineEdit::Normal, "", &ok);
        if (!ok || word.isEmpty()) return;
        word = word.toLower().trimmed();

        for (const QString& file : files)
        {
            QImage img(file);
            if (!img.isNull()) addTrainingPair(file, img, word);
        }
        updateSampleList();
    }

    void onStartTraining()
    {
        if (samples.empty())
        {
            QMessageBox::warning(this, "No Data", "Add training pairs first.");
            return;
        }

        provider.setTrainingData(samples);
        ann->setTrainer(&provider);

        if (worker) { worker->requestStop(); worker->wait(); delete worker; }
        worker = new TrainingWorker(this);
        worker->setNetwork(ann);
        worker->setIterations(spinIters->value());

        connect(worker, &TrainingWorker::progressUpdated,
                this, &VisionBrainApp::onProgress);
        connect(worker, &TrainingWorker::finished,
                this, &VisionBrainApp::onTrainDone);
        connect(worker, &TrainingWorker::errorOccurred,
                this, [this](const QString& e) {
            QMessageBox::critical(this, "Error", e);
            btnTrain->setEnabled(true); btnStop->setEnabled(false);
        });

        btnTrain->setEnabled(false);
        btnStop->setEnabled(true);
        progressBar->setValue(0);
        log("Training started: " + QString::number(spinIters->value()) +
            " iters, " + QString::number(samples.size()) + " samples");
        worker->start();
    }

    void onStopTraining()
    {
        if (worker) worker->requestStop();
        btnTrain->setEnabled(true);
        btnStop->setEnabled(false);
        log("Training stopped.");
    }

    void onProgress(int cur, int tot)
    {
        progressBar->setMaximum(tot);
        progressBar->setValue(cur);
        lblStatus->setText(QString("Iter %1/%2").arg(cur).arg(tot));
    }

    void onTrainDone()
    {
        btnTrain->setEnabled(true);
        btnStop->setEnabled(false);
        progressBar->setValue(progressBar->maximum());
        lblStatus->setText("Complete!");
        log("Training complete.");

        QSqlQuery q(dbMgr.getDb());
        q.prepare("INSERT INTO training_sessions (session_name, iterations, samples) VALUES (?,?,?)");
        q.addBindValue(QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss"));
        q.addBindValue(spinIters->value());
        q.addBindValue((int)samples.size());
        q.exec();
    }

    void onSaveMap()
    {
        QString f = QFileDialog::getSaveFileName(this, "Save Neural Map", "", "*.nmap");
        if (f.isEmpty()) return;
        ann->saveNeuralMap(f.toStdString().c_str());
        log("Saved: " + f);
        QSqlQuery q(dbMgr.getDb());
        q.prepare("INSERT INTO neural_maps (name, filepath) VALUES (?,?)");
        q.addBindValue(QFileInfo(f).baseName());
        q.addBindValue(f);
        q.exec();
    }

    void onLoadMap()
    {
        QString f = QFileDialog::getOpenFileName(this, "Load Neural Map", "", "*.nmap");
        if (f.isEmpty()) return;
        if (ann->loadNeuralMap(f.toStdString().c_str()))
            log("Loaded: " + f);
        else
            QMessageBox::critical(this, "Error", "Load failed - architecture mismatch.");
    }

    void onTestImage()
    {
        QString f = QFileDialog::getOpenFileName(this, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)");
        if (f.isEmpty()) return;
        QImage img(f);
        if (img.isNull()) return;

        lblTestImg->setPixmap(QPixmap::fromImage(
            img.scaled(256, 256, Qt::KeepAspectRatio, Qt::SmoothTransformation)));

        std::vector<float> px = ImageProcessor::imageToVector(img);
        ann->setInput(px);
        std::vector<float> out = ann->getOutput();

        float conf = 0;
        QString word = vocabulary.decodeOutput(out, &conf);

        // Build response
        QString response;
        if (conf > 0.8f)
            response = "I clearly see: " + word + "! (confidence: " +
                       QString::number((int)(conf * 100)) + "%)";
        else if (conf > 0.5f)
            response = "I think it's: " + word + " (confidence: " +
                       QString::number((int)(conf * 100)) + "%)";
        else
            response = "Might be: " + word + "? (low confidence: " +
                       QString::number((int)(conf * 100)) + "%)";

        // Top-3 analysis
        response += "\n\nTop matches:\n";
        std::vector<std::pair<float, int>> scored;
        for (int i = 0; i < (int)out.size() && i < vocabulary.size(); ++i)
            scored.push_back({out[i], i});
        std::sort(scored.begin(), scored.end(),
                  [](const std::pair<float,int>& a, const std::pair<float,int>& b)
                  { return a.first > b.first; });
        for (int i = 0; i < qMin(3, (int)scored.size()); ++i)
            response += "  " + vocabulary.getWord(scored[i].second) + " (" +
                        QString::number((int)(scored[i].first * 100)) + "%)\n";

        txtResponse->setPlainText(response);

        QSqlQuery q(dbMgr.getDb());
        q.prepare("INSERT INTO conversation_log (input_image, response, confidence) VALUES (?,?,?)");
        q.addBindValue(f);
        q.addBindValue(word);
        q.addBindValue(conf);
        q.exec();

        log("Test: " + QFileInfo(f).fileName() + " -> " + word);
    }

    void onAddFusionMap()
    {
        QStringList files = QFileDialog::getOpenFileNames(this, "Add Maps", "", "*.nmap");
        for (const QString& f : files)
            fusionList->addItem(f);
    }

    void onRemoveFusionMap()
    {
        qDeleteAll(fusionList->selectedItems());
    }

    void onFuse()
    {
        if (fusionList->count() < 2)
        {
            QMessageBox::warning(this, "Fusion", "Need at least 2 maps.");
            return;
        }
        QString out = QFileDialog::getSaveFileName(this, "Save Fused Brain", "", "*.nmap");
        if (out.isEmpty()) return;

        QStringList maps;
        for (int i = 0; i < fusionList->count(); ++i)
            maps.append(fusionList->item(i)->text());

        auto method = (BrainFusionEngine::FusionMethod)cmbMethod->currentIndex();
        if (BrainFusionEngine::fuseNeuralMaps(maps, out, method))
        {
            log("Fused " + QString::number(maps.size()) + " maps -> " + out);
            QMessageBox::information(this, "Success", "Brain fusion complete!");
            QSqlQuery q(dbMgr.getDb());
            q.prepare("INSERT INTO brain_fusions (name, source_maps, output_map, method) VALUES (?,?,?,?)");
            q.addBindValue(QFileInfo(out).baseName());
            q.addBindValue(maps.join(";"));
            q.addBindValue(out);
            q.addBindValue(cmbMethod->currentText());
            q.exec();
        }
        else
            QMessageBox::critical(this, "Error", "Fusion failed.");
    }

    void onRefreshDb()
    {
        QSqlQuery q(dbMgr.getDb());
        q.exec("SELECT filepath, description, created_at FROM images ORDER BY id DESC LIMIT 200");
        dbTable->setRowCount(0);
        int row = 0;
        while (q.next())
        {
            dbTable->insertRow(row);
            dbTable->setItem(row, 0, new QTableWidgetItem(q.value(0).toString()));
            dbTable->setItem(row, 1, new QTableWidgetItem(q.value(1).toString()));
            dbTable->setItem(row, 2, new QTableWidgetItem(q.value(2).toString()));
            row++;
        }
        // Vocab list
        vocabList->clear();
        for (int i = 0; i < vocabulary.size(); ++i)
            vocabList->addItem("[" + QString::number(i) + "] " + vocabulary.getWord(i));
    }

    void onClearData()
    {
        if (QMessageBox::question(this, "Confirm", "Clear all training data?") == QMessageBox::Yes)
        {
            samples.clear();
            updateSampleList();
        }
    }

    void onExportVocab()
    {
        QString f = QFileDialog::getSaveFileName(this, "Export", "", "*.txt");
        if (f.isEmpty()) return;
        QFile file(f);
        if (file.open(QIODevice::WriteOnly | QIODevice::Text))
        {
            QTextStream ts(&file);
            for (int i = 0; i < vocabulary.size(); ++i)
                ts << vocabulary.getWord(i) << "\n";
            file.close();
            log("Vocabulary exported: " + f);
        }
    }

    void onImportVocab()
    {
        QString f = QFileDialog::getOpenFileName(this, "Import", "", "*.txt");
        if (f.isEmpty()) return;
        QFile file(f);
        if (file.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            QTextStream ts(&file);
            while (!ts.atEnd())
            {
                QString w = ts.readLine().trimmed();
                if (!w.isEmpty()) vocabulary.addWord(w);
            }
            file.close();
            log("Vocabulary imported. Size: " + QString::number(vocabulary.size()));
        }
    }

private:
    void addTrainingPair(const QString& path, const QImage& img, const QString& word)
    {
        int wIdx = vocabulary.addWord(word);
        if (wIdx < 0) return;

        TrainingSample s;
        s.input = ImageProcessor::imageToVector(img);
        s.output = vocabulary.encodeWord(word);
        s.word = word;
        s.imagePath = path;
        samples.push_back(s);

        QSqlQuery q(dbMgr.getDb());
        q.prepare("INSERT INTO images (filepath, description) VALUES (?,?)");
        q.addBindValue(path); q.addBindValue(word); q.exec();
        int imgId = q.lastInsertId().toInt();

        q.prepare("INSERT OR IGNORE INTO words (word) VALUES (?)");
        q.addBindValue(word); q.exec();
        q.prepare("SELECT id FROM words WHERE word=?");
        q.addBindValue(word); q.exec();
        int wId = q.next() ? q.value(0).toInt() : -1;

        if (imgId > 0 && wId > 0)
        {
            q.prepare("INSERT INTO image_word_pairs (image_id, word_id) VALUES (?,?)");
            q.addBindValue(imgId); q.addBindValue(wId); q.exec();
        }
    }

    void updateSampleList()
    {
        sampleList->clear();
        for (const auto& s : samples)
            sampleList->addItem(QFileInfo(s.imagePath).fileName() + " -> \"" + s.word + "\"");
        lblStatus->setText(QString("%1 samples, %2 words").arg(samples.size()).arg(vocabulary.size()));
    }

    void setupUi()
    {
        QWidget* central = new QWidget;
        setCentralWidget(central);
        QVBoxLayout* main = new QVBoxLayout(central);
        tabs = new QTabWidget;

        // --- TRAINING TAB ---
        QWidget* t1 = new QWidget;
        QVBoxLayout* l1 = new QVBoxLayout(t1);

        QHBoxLayout* btnRow = new QHBoxLayout;
        QPushButton* bLoad = new QPushButton("Load & Label Images");
        QPushButton* bBatch = new QPushButton("Batch Label");
        QPushButton* bClear = new QPushButton("Clear All");
        btnRow->addWidget(bLoad); btnRow->addWidget(bBatch);
        btnRow->addWidget(bClear); btnRow->addStretch();
        l1->addLayout(btnRow);

        sampleList = new QListWidget;
        l1->addWidget(sampleList);

        QGridLayout* grid = new QGridLayout;
        grid->addWidget(new QLabel("Iterations:"), 0, 0);
        spinIters = new QSpinBox; spinIters->setRange(100, 1000000);
        spinIters->setValue(5000); spinIters->setSingleStep(500);
        grid->addWidget(spinIters, 0, 1);

        btnTrain = new QPushButton("Start Training");
        btnStop = new QPushButton("Stop"); btnStop->setEnabled(false);
        grid->addWidget(btnTrain, 1, 0); grid->addWidget(btnStop, 1, 1);

        progressBar = new QProgressBar;
        grid->addWidget(progressBar, 2, 0, 1, 2);
        lblStatus = new QLabel("Idle");
        grid->addWidget(lblStatus, 3, 0, 1, 2);
        l1->addLayout(grid);

        QHBoxLayout* ioRow = new QHBoxLayout;
        QPushButton* bSave = new QPushButton("Save Neural Map");
        QPushButton* bLoadN = new QPushButton("Load Neural Map");
        ioRow->addWidget(bSave); ioRow->addWidget(bLoadN); ioRow->addStretch();
        l1->addLayout(ioRow);

        tabs->addTab(t1, "Training");

        // --- VISION TAB ---
        QWidget* t2 = new QWidget;
        QHBoxLayout* l2 = new QHBoxLayout(t2);
        QVBoxLayout* left = new QVBoxLayout;
        QPushButton* bTest = new QPushButton("Analyze Image");
        left->addWidget(bTest);
        lblTestImg = new QLabel("No image");
        lblTestImg->setMinimumSize(280, 280);
        lblTestImg->setAlignment(Qt::AlignCenter);
        lblTestImg->setStyleSheet("background:#1a1a2e;border:1px solid #444;border-radius:6px;");
        left->addWidget(lblTestImg);
        left->addStretch();
        l2->addLayout(left);

        QVBoxLayout* right = new QVBoxLayout;
        right->addWidget(new QLabel("Brain Response:"));
        txtResponse = new QPlainTextEdit;
        txtResponse->setReadOnly(true);
        txtResponse->setFont(QFont("Monospace", 11));
        right->addWidget(txtResponse);
        l2->addLayout(right);
        tabs->addTab(t2, "Vision & Talk");

        // --- FUSION TAB ---
        QWidget* t3 = new QWidget;
        QVBoxLayout* l3 = new QVBoxLayout(t3);
        QGroupBox* fg = new QGroupBox("Brain Fusion - Combine Neural Maps");
        QVBoxLayout* fl = new QVBoxLayout(fg);
        QHBoxLayout* fb = new QHBoxLayout;
        QPushButton* bAdd = new QPushButton("Add Maps");
        QPushButton* bRem = new QPushButton("Remove");
        fb->addWidget(bAdd); fb->addWidget(bRem); fb->addStretch();
        fl->addLayout(fb);
        fusionList = new QListWidget;
        fl->addWidget(fusionList);
        QHBoxLayout* fm = new QHBoxLayout;
        fm->addWidget(new QLabel("Method:"));
        cmbMethod = new QComboBox;
        cmbMethod->addItems({"Average", "Weighted Avg", "Max Activation", "Median"});
        fm->addWidget(cmbMethod); fm->addStretch();
        fl->addLayout(fm);
        QPushButton* bFuse = new QPushButton("FUSE BRAINS");
        bFuse->setMinimumHeight(44);
        bFuse->setStyleSheet("QPushButton{background:#27ae60;color:#fff;font-weight:bold;"
                             "font-size:14px;border-radius:6px}"
                             "QPushButton:hover{background:#2ecc71}");
        fl->addWidget(bFuse);
        l3->addWidget(fg);
        tabs->addTab(t3, "Brain Fusion");

        // --- DATABASE TAB ---
        QWidget* t4 = new QWidget;
        QVBoxLayout* l4 = new QVBoxLayout(t4);
        QHBoxLayout* dbBtns = new QHBoxLayout;
        QPushButton* bRef = new QPushButton("Refresh");
        QPushButton* bExpV = new QPushButton("Export Vocab");
        QPushButton* bImpV = new QPushButton("Import Vocab");
        dbBtns->addWidget(bRef); dbBtns->addWidget(bExpV);
        dbBtns->addWidget(bImpV); dbBtns->addStretch();
        l4->addLayout(dbBtns);
        dbTable = new QTableWidget(0, 3);
        dbTable->setHorizontalHeaderLabels({"File", "Label", "Date"});
        dbTable->horizontalHeader()->setStretchLastSection(true);
        l4->addWidget(dbTable);
        vocabList = new QListWidget;
        vocabList->setMaximumHeight(150);
        l4->addWidget(vocabList);
        tabs->addTab(t4, "Database");

        // --- LOG TAB ---
        QWidget* t5 = new QWidget;
        QVBoxLayout* l5 = new QVBoxLayout(t5);
        txtLog = new QTextEdit; txtLog->setReadOnly(true);
        txtLog->setFont(QFont("Monospace", 9));
        l5->addWidget(txtLog);
        tabs->addTab(t5, "Log");

        main->addWidget(tabs);

        // Connections
        connect(bLoad, &QPushButton::clicked, this, &VisionBrainApp::onLoadImages);
        connect(bBatch, &QPushButton::clicked, this, &VisionBrainApp::onBatchLabel);
        connect(bClear, &QPushButton::clicked, this, &VisionBrainApp::onClearData);
        connect(btnTrain, &QPushButton::clicked, this, &VisionBrainApp::onStartTraining);
        connect(btnStop, &QPushButton::clicked, this, &VisionBrainApp::onStopTraining);
        connect(bSave, &QPushButton::clicked, this, &VisionBrainApp::onSaveMap);
        connect(bLoadN, &QPushButton::clicked, this, &VisionBrainApp::onLoadMap);
        connect(bTest, &QPushButton::clicked, this, &VisionBrainApp::onTestImage);
        connect(bAdd, &QPushButton::clicked, this, &VisionBrainApp::onAddFusionMap);
        connect(bRem, &QPushButton::clicked, this, &VisionBrainApp::onRemoveFusionMap);
        connect(bFuse, &QPushButton::clicked, this, &VisionBrainApp::onFuse);
        connect(bRef, &QPushButton::clicked, this, &VisionBrainApp::onRefreshDb);
        connect(bExpV, &QPushButton::clicked, this, &VisionBrainApp::onExportVocab);
        connect(bImpV, &QPushButton::clicked, this, &VisionBrainApp::onImportVocab);
    }

    void setupDatabase()
    {
        QString path = QDir::currentPath() + "/vision_brain.db";
        if (dbMgr.open(path))
            log("Database: " + path);
        else
            log("WARNING: Could not open database!");
    }

    void setupMenu()
    {
        QMenu* fm = menuBar()->addMenu("&File");
        fm->addAction("Save Network", this, &VisionBrainApp::onSaveMap);
        fm->addAction("Load Network", this, &VisionBrainApp::onLoadMap);
        fm->addSeparator();
        fm->addAction("Exit", this, &QWidget::close);
        QMenu* hm = menuBar()->addMenu("&Help");
        hm->addAction("About", [this]() {
            QMessageBox::about(this, "Vision Brain",
                "Vision-to-Speech Neural Learning Studio\n\n"
                "Architecture: 256 -> 64 -> 64 -> 128\n"
                "Train with image-word pairs, fuse brains,\n"
                "and let it talk about what it sees.");
        });
    }

    void log(const QString& msg)
    {
        txtLog->append("[" + QTime::currentTime().toString("hh:mm:ss") + "] " + msg);
    }

    // Members
    QTabWidget* tabs;
    Ann* ann;
    Vocabulary vocabulary;
    DatabaseManager dbMgr;
    TrainingDataProvider provider;
    TrainingWorker* worker;
    std::vector<TrainingSample> samples;

    QListWidget* sampleList;
    QSpinBox* spinIters;
    QPushButton* btnTrain;
    QPushButton* btnStop;
    QProgressBar* progressBar;
    QLabel* lblStatus;
    QLabel* lblTestImg;
    QPlainTextEdit* txtResponse;
    QListWidget* fusionList;
    QComboBox* cmbMethod;
    QTableWidget* dbTable;
    QListWidget* vocabList;
    QTextEdit* txtLog;
};

// ============================================================
// MAIN
// ============================================================

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    app.setApplicationName("VisionBrain");
    app.setStyle("Fusion");

    QPalette pal;
    pal.setColor(QPalette::Window, QColor(45, 45, 48));
    pal.setColor(QPalette::WindowText, Qt::white);
    pal.setColor(QPalette::Base, QColor(30, 30, 32));
    pal.setColor(QPalette::AlternateBase, QColor(45, 45, 48));
    pal.setColor(QPalette::Text, Qt::white);
    pal.setColor(QPalette::Button, QColor(53, 53, 56));
    pal.setColor(QPalette::ButtonText, Qt::white);
    pal.setColor(QPalette::Highlight, QColor(42, 130, 218));
    pal.setColor(QPalette::HighlightedText, Qt::white);
    app.setPalette(pal);

    VisionBrainApp win;
    win.show();
    return app.exec();
}

#include "main.moc"
