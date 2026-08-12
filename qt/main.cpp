// ============================================================================
//  AgentMind — hybrid multi-agent Markov brain with SQLite memory (Qt 5.12)
//
//  * Hybrid Markov chain:  trigram -> bigram -> unigram fallback sampling
//  * 4 agents (Librarian / Analyst / Echo / Dreamer) + weighted Director
//  * SQLite long-term memory: tokens, transitions, mood affinities, sentences,
//    entities, episodes (chat log), key-value state  -> survives restarts
//  * Pronoun resolution (it/they/that/...) against recent entities
//  * Mood detection + mood forcing biases word sampling
//  * Self-learning: every user message trains the chain; thumbs up/down
//    reinforce/decay the exact transitions used in the last answer
// ============================================================================
#if defined(_MSC_VER)
#pragma execution_character_set("utf-8")
#endif

#include <QApplication>
#include <QMainWindow>
#include <QSplitter>
#include <QTabWidget>
#include <QTextBrowser>
#include <QLineEdit>
#include <QPushButton>
#include <QComboBox>
#include <QSlider>
#include <QLabel>
#include <QListWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QStatusBar>
#include <QToolBar>
#include <QAction>
#include <QTextCursor>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QSqlError>
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QStandardPaths>
#include <QDateTime>
#include <QJsonDocument>
#include <QJsonArray>
#include <QRegularExpression>
#include <QSet>
#include <QMap>
#include <QElapsedTimer>
#include <QtGlobal>

#include <random>
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// Moods + lexicon
// ---------------------------------------------------------------------------
enum Mood { MoodNone = 0, MoodJoy = 1, MoodSorrow = 2, MoodCuriosity = 3,
            MoodFear = 4, MoodAnger = 5, MoodCalm = 6 };

static QString moodName(int m)
{
    switch (m) {
    case MoodJoy:       return "joyful";
    case MoodSorrow:    return "melancholic";
    case MoodCuriosity: return "curious";
    case MoodFear:      return "fearful";
    case MoodAnger:     return "angry";
    case MoodCalm:      return "calm";
    default:            return "neutral";
    }
}

static const QHash<QString, int> &moodLex()
{
    static QHash<QString, int> lex = {
        // joy
        {"happy",MoodJoy},{"joy",MoodJoy},{"joyful",MoodJoy},{"laugh",MoodJoy},
        {"laughter",MoodJoy},{"smile",MoodJoy},{"smiling",MoodJoy},{"bright",MoodJoy},
        {"wonderful",MoodJoy},{"great",MoodJoy},{"delight",MoodJoy},{"fun",MoodJoy},
        {"love",MoodJoy},{"loved",MoodJoy},{"lovely",MoodJoy},{"glad",MoodJoy},
        {"cheerful",MoodJoy},{"shine",MoodJoy},
        // sorrow
        {"sad",MoodSorrow},{"sorrow",MoodSorrow},{"cry",MoodSorrow},{"tears",MoodSorrow},
        {"tear",MoodSorrow},{"gloom",MoodSorrow},{"gloomy",MoodSorrow},{"melancholy",MoodSorrow},
        {"lost",MoodSorrow},{"grief",MoodSorrow},{"lonely",MoodSorrow},{"mourn",MoodSorrow},
        {"grey",MoodSorrow},{"dreary",MoodSorrow},
        // curiosity
        {"curious",MoodCuriosity},{"curiosity",MoodCuriosity},{"wonder",MoodCuriosity},
        {"question",MoodCuriosity},{"questions",MoodCuriosity},{"why",MoodCuriosity},
        {"seek",MoodCuriosity},{"discover",MoodCuriosity},{"perhaps",MoodCuriosity},
        {"mystery",MoodCuriosity},{"puzzle",MoodCuriosity},{"explore",MoodCuriosity},
        {"strange",MoodCuriosity},
        // fear
        {"fear",MoodFear},{"afraid",MoodFear},{"dread",MoodFear},{"terror",MoodFear},
        {"shadow",MoodFear},{"shadows",MoodFear},{"tremble",MoodFear},{"worry",MoodFear},
        {"worried",MoodFear},{"anxious",MoodFear},{"fright",MoodFear},{"scary",MoodFear},
        // anger
        {"anger",MoodAnger},{"angry",MoodAnger},{"rage",MoodAnger},{"fury",MoodAnger},
        {"furious",MoodAnger},{"hate",MoodAnger},{"hatred",MoodAnger},{"fierce",MoodAnger},
        {"wrath",MoodAnger},{"burn",MoodAnger},{"burning",MoodAnger},{"mad",MoodAnger},
        // calm
        {"calm",MoodCalm},{"quiet",MoodCalm},{"still",MoodCalm},{"peaceful",MoodCalm},
        {"peace",MoodCalm},{"gentle",MoodCalm},{"softly",MoodCalm},{"slow",MoodCalm},
        {"silence",MoodCalm},{"silent",MoodCalm},{"rest",MoodCalm},{"breathe",MoodCalm},
        {"ease",MoodCalm}
    };
    return lex;
}

static const QSet<QString> &stopWords()
{
    static QSet<QString> s = {
        "the","a","an","and","or","but","if","then","else","of","to","in","on","at",
        "for","with","as","by","is","are","was","were","be","been","being","it","its",
        "this","that","these","those","i","you","he","she","we","they","them","his",
        "her","their","our","your","my","me","him","us","do","does","did","not","no",
        "so","too","very","just","about","into","over","after","before","from","up",
        "down","out","what","when","where","who","why","how","can","could","would",
        "should","will","shall","may","might","must","there","here","all","any",
        "some","one","two","also","than","more","most","such","only","own","same",
        "am","has","have","had","having","and","like","get","got","yes","well"
    };
    return s;
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------
struct Tok { QString orig; QString lower; bool cap = false; };

static QVector<QVector<Tok>> splitSentences(const QString &text)
{
    QVector<QVector<Tok>> out;
    static const QRegularExpression sentRe("(?:[.!?]+[\"')\\]]*\\s+|\\n+)");
    static const QRegularExpression wordRe("[A-Za-z][A-Za-z'\\-]*|\\d+(?:\\.\\d+)?");
    const QStringList parts = text.split(sentRe, QString::SkipEmptyParts);
    for (const QString &p : parts) {
        QVector<Tok> s;
        QRegularExpressionMatchIterator it = wordRe.globalMatch(p);
        while (it.hasNext()) {
            QRegularExpressionMatch m = it.next();
            QString w = m.captured();
            if (w.length() > 24) w = w.left(24);
            Tok t;
            t.orig  = w;
            t.lower = w.toLower();
            t.cap   = !w.isEmpty() && w.at(0).isUpper();
            s.append(t);
        }
        if (!s.isEmpty()) out.append(s);
    }
    return out;
}

static QString htmlEscape(const QString &s) { return s.toHtmlEscaped(); }

// ---------------------------------------------------------------------------
// Brain — hybrid Markov engine + multi-agent director + SQLite memory
// ---------------------------------------------------------------------------
class Brain
{
public:
    struct Reply {
        QString text;
        QString agent = "Director";
        QString moodName = "neutral";
        QStringList notes;
        QVector<int> path;          // token ids of the answer (for reinforcement)
    };
    struct Stats { int tokens=0, transitions=0, sentences=0, episodes=0, entities=0; };
    struct LearnStats { int sentences=0, tokens=0, entities=0; qint64 ms=0; };
    struct Episode { QString role, content, meta; };

    QStringList recentEntities;     // most recent first; pronouns resolve here
    QSet<QString> mutedAgents;
    int forcedMood = MoodNone;      // 0 = auto-detect
    double temperature = 1.0;

    bool open(const QString &path, QString *err)
    {
        m_path = path;
        db = QSqlDatabase::addDatabase("QSQLITE", "agentmind");
        db.setDatabaseName(path);
        if (!db.open()) { if (err) *err = db.lastError().text(); return false; }
        db.exec("PRAGMA journal_mode=WAL");
        db.exec("PRAGMA synchronous=NORMAL");
        const char *schema =
            "CREATE TABLE IF NOT EXISTS tokens(id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " word TEXT UNIQUE NOT NULL, freq REAL DEFAULT 0, ends REAL DEFAULT 0);"
            "CREATE TABLE IF NOT EXISTS trans(id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " a INTEGER NOT NULL, b INTEGER NOT NULL, c INTEGER NOT NULL,"
            " w REAL DEFAULT 1, UNIQUE(a,b,c));"
            "CREATE INDEX IF NOT EXISTS idx_trans_ab ON trans(a,b);"
            "CREATE INDEX IF NOT EXISTS idx_trans_b ON trans(b);"
            "CREATE TABLE IF NOT EXISTS tmoods(token INTEGER NOT NULL,"
            " mood INTEGER NOT NULL, w REAL DEFAULT 0, UNIQUE(token,mood));"
            "CREATE TABLE IF NOT EXISTS sentences(id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " text TEXT NOT NULL, source TEXT);"
            "CREATE INDEX IF NOT EXISTS idx_sent_src ON sentences(source);"
            "CREATE TABLE IF NOT EXISTS episodes(id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " role TEXT, content TEXT, meta TEXT, ts TEXT);"
            "CREATE TABLE IF NOT EXISTS entities(id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " name TEXT UNIQUE NOT NULL, mentions INTEGER DEFAULT 1);"
            "CREATE TABLE IF NOT EXISTS kv(k TEXT UNIQUE NOT NULL, v TEXT);";
        if (!db.exec(schema).isActive() && db.lastError().isValid()) {
            if (err) *err = db.lastError().text();
            return false;
        }
        loadCache();
        loadRecents();
        if (m_words.isEmpty()) {          // first boot: tiny seed corpus
            learnTextCore(seedCorpus(), "seed-corpus", MoodNone, 500, true, false);
            recentEntities = QStringList{ "Ada", "the lighthouse keeper" };
            persistRecents();
        }
        return true;
    }

    // ---------------- learning ----------------
    LearnStats learnFile(const QString &text, const QString &fileName)
    {
        LearnStats st;
        QElapsedTimer t; t.start();
        m_newEntities = 0; m_lastSentCount = 0; m_lastTokCount = 0;
        learnTextCore(text, fileName, MoodNone, 6000, true, false);
        kvSet("current_source", fileName);
        pushRecent(QFileInfo(fileName).baseName());
        QStringList srcs = kvGet("sources").split('|', QString::SkipEmptyParts);
        if (!srcs.contains(fileName)) {
            srcs.prepend(fileName);
            while (srcs.size() > 15) srcs.removeLast();
            kvSet("sources", srcs.join('|'));
        }
        persistRecents();
        st.sentences = m_lastSentCount; st.tokens = m_lastTokCount;
        st.entities = m_newEntities; st.ms = t.elapsed();
        return st;
    }

    // ---------------- conversation ----------------
    Reply respond(const QString &userText)
    {
        Reply R;
        QVector<QVector<Tok>> sents = splitSentences(userText);
        QVector<Tok> flat;
        for (const QVector<Tok> &s : sents) flat += s;
        QStringList lowers;
        for (const Tok &t : flat) lowers << t.lower;

        // forget command
        if (userText.contains("forget", Qt::CaseInsensitive) &&
            (userText.contains("conversation") || userText.contains("chat") ||
             userText.contains("everything"))) {
            forgetConversation();
            R.agent = "Director";
            R.text = "Done. Our chat log is wiped — but everything I read is still inside me.";
            QSqlQuery q(db);
            q.prepare("INSERT INTO episodes(role,content,meta,ts) VALUES('bot',?,?,?)");
            q.addBindValue(R.text); q.addBindValue("Director|memory wipe");
            q.addBindValue(QDateTime::currentDateTime().toString(Qt::ISODate));
            q.exec();
            return R;
        }

        // entities mentioned by the user (capitalized runs) -> recents
        for (const QVector<Tok> &s : sents) noteEntitiesInSentence(s, true);

        int userMood = detectMood(lowers);
        int mood = (forcedMood != MoodNone) ? forcedMood : userMood;
        R.moodName = moodName(mood);

        // pronoun resolution
        QStringList pronounNotes;
        QStringList resolved = resolvePronouns(lowers, pronounNotes);
        R.notes += pronounNotes;

        // self-learning: the user's own words train the chain
        learnTextCore(userText, "conversation", userMood, 40, false, false);

        bool isQuestion = userText.contains('?') || startsWithQuestionWord(lowers);
        bool isAck = (lowers.size() <= 2);

        QVector<int> seedIds;
        for (const QString &w : resolved) {
            QHash<QString,int>::ConstIterator it = m_wordId.find(w);
            if (it != m_wordId.end()) seedIds.append(it.value());
        }

        // ---- the four agents propose ----
        Cand lib = librarianReply(resolved);
        Cand ana = analystReply(seedIds, mood, isAck);
        Cand eko = echoReply(seedIds, mood, userMood);
        Cand drm = dreamerReply(mood);

        // ---- Director: weighted selection + occasional blending ----
        struct W { Cand c; double w; };
        QVector<W> pool;
        auto consider = [&](const Cand &c, double base) {
            if (c.text.isEmpty() || mutedAgents.contains(c.agent)) return;
            double m = 1.0;
            if (isQuestion) {
                if (c.agent == "Librarian") m *= 1.9;
                if (c.agent == "Analyst")   m *= 1.2;
            }
            if (userMood != MoodNone && c.agent == "Echo") m *= 1.6;
            if (forcedMood != MoodNone && (c.agent == "Echo" || c.agent == "Dreamer")) m *= 1.25;
            if (isAck && c.agent == "Analyst") m *= 1.7;
            m *= agentBoost(c.agent);
            m *= 0.85 + 0.3 * uniform01();
            pool.append({ c, base * c.score * m });
        };
        consider(lib, 1.05); consider(ana, 1.0); consider(eko, 0.95); consider(drm, 0.80);

        Cand winner;
        if (pool.isEmpty()) {
            winner.agent = "Director";
            winner.text = m_words.isEmpty()
                ? "My memory is completely empty. Load a text file with 'Read files…' and I will learn it."
                : "I am not sure what to say — ask me about something I have read.";
        } else {
            QVector<QPair<int,double>> wp;
            for (int i = 0; i < pool.size(); ++i) wp.append({ i, pool[i].w });
            int idx = weightedPick(wp);
            winner = pool[idx].c;
            if (pool.size() > 1 && uniform01() < 0.25) {   // hybrid blend
                QVector<QPair<int,double>> rest;
                for (int i = 0; i < pool.size(); ++i) if (i != idx) rest.append({ i, pool[i].w });
                int j = weightedPick(rest);
                const Cand &second = pool[j].c;
                if (!second.text.isEmpty() && second.agent != winner.agent &&
                    winner.text.length() + second.text.length() < 420) {
                    winner.text += " " + second.text;
                    R.notes << ("blended with " + second.agent);
                }
            }
        }

        // tiny intents
        if (!lowers.isEmpty() && lowers.size() <= 4) {
            const QString f = lowers.first();
            if (f == "hi" || f == "hello" || f == "hey")
                winner.text = "Hello! " + winner.text;
        }
        if (userText.contains("who are you") || userText.contains("your name")) {
            winner.agent = "Director"; winner.path.clear();
            winner.text = "I am AgentMind: four little Markov agents — Librarian, Analyst, "
                          "Echo and Dreamer — sharing one SQLite memory. Feed me text and "
                          "I will talk about it.";
        }
        if (lowers.contains("help"))
            R.notes << "tip: load files, use pronouns (it/they/that), set a mood, rate answers with thumbs";

        R.text = winner.text.left(600);
        R.agent = winner.agent;
        R.path = winner.path;
        m_lastPath = winner.path;
        m_lastAgent = winner.agent;

        // persist episodes
        QString ts = QDateTime::currentDateTime().toString(Qt::ISODate);
        QSqlQuery q(db);
        q.prepare("INSERT INTO episodes(role,content,meta,ts) VALUES('user',?,?,?)");
        q.addBindValue(userText); q.addBindValue(""); q.addBindValue(ts); q.exec();
        q.prepare("INSERT INTO episodes(role,content,meta,ts) VALUES('bot',?,?,?)");
        q.addBindValue(R.text);
        q.addBindValue(winner.agent + "|" + R.notes.join(" ; "));
        q.addBindValue(ts); q.exec();

        persistRecents();
        if (mood != MoodNone) kvSet("last_mood", QString::number(mood));
        return R;
    }

    // reinforcement learning on the last answer's exact transitions
    bool feedback(bool positive)
    {
        if (m_lastPath.size() < 2) return false;
        double mult = positive ? 1.3 : 0.55;
        QSqlQuery q(db);
        q.prepare("UPDATE trans SET w=MAX(0.05, w*?) WHERE a=? AND b=? AND c=?");
        db.transaction();
        for (int i = 1; i < m_lastPath.size(); ++i) {
            q.bindValue(0, mult); q.bindValue(1, -1);
            q.bindValue(2, m_lastPath[i-1]); q.bindValue(3, m_lastPath[i]);
            q.exec();
        }
        for (int i = 2; i < m_lastPath.size(); ++i) {
            q.bindValue(0, mult); q.bindValue(1, m_lastPath[i-2]);
            q.bindValue(2, m_lastPath[i-1]); q.bindValue(3, m_lastPath[i]);
            q.exec();
        }
        db.commit();
        double b = kvGet("boost_" + m_lastAgent, "1.0").toDouble();
        b = qBound(0.2, b + (positive ? 0.2 : -0.25), 3.0);
        kvSet("boost_" + m_lastAgent, QString::number(b, 'f', 2));
        QString key = positive ? "praise" : "criticism";
        kvSet(key, QString::number(kvGet(key, "0").toInt() + 1));
        return true;
    }

    QString lastAgent() const { return m_lastAgent; }

    // ---------------- memory ops ----------------
    void forgetConversation()
    {
        db.exec("DELETE FROM episodes");
        m_lastPath.clear(); m_lastAgent.clear();
    }

    void wipe()
    {
        db.close();
        db = QSqlDatabase();
        QSqlDatabase::removeDatabase("agentmind");
        QFile::remove(m_path); QFile::remove(m_path + "-wal"); QFile::remove(m_path + "-shm");
        QString err;
        open(m_path, &err);
    }

    Stats stats()
    {
        Stats s;
        auto cnt = [&](const char *t) {
            QSqlQuery q(db);
            q.exec(QString("SELECT COUNT(*) FROM ") + t);
            return q.next() ? q.value(0).toInt() : 0;
        };
        s.tokens = cnt("tokens"); s.transitions = cnt("trans");
        s.sentences = cnt("sentences"); s.episodes = cnt("episodes");
        s.entities = cnt("entities");
        return s;
    }

    QStringList topEntities(int n)
    {
        QStringList out;
        QSqlQuery q(db);
        q.prepare("SELECT name,mentions FROM entities ORDER BY mentions DESC, id DESC LIMIT ?");
        q.addBindValue(n);
        if (q.exec()) while (q.next())
            out << QString("%1 (%2)").arg(q.value(0).toString()).arg(q.value(1).toInt());
        return out;
    }

    QVector<Episode> episodes()
    {
        QVector<Episode> out;
        QSqlQuery q(db);
        if (q.exec("SELECT role,content,meta FROM episodes ORDER BY id"))
            while (q.next())
                out.append({ q.value(0).toString(), q.value(1).toString(), q.value(2).toString() });
        return out;
    }

    QString kvGet(const QString &k, const QString &def = QString())
    {
        QSqlQuery q(db);
        q.prepare("SELECT v FROM kv WHERE k=?"); q.addBindValue(k);
        if (q.exec() && q.next()) return q.value(0).toString();
        return def;
    }
    void kvSet(const QString &k, const QString &v)
    {
        QSqlQuery q(db);
        q.prepare("UPDATE kv SET v=? WHERE k=?");
        q.addBindValue(v); q.addBindValue(k);
        if (q.exec() && q.numRowsAffected() == 0) {
            q.prepare("INSERT INTO kv(k,v) VALUES(?,?)");
            q.addBindValue(k); q.addBindValue(v); q.exec();
        }
    }

private:
    struct Cand { QString agent; QString text; double score = 0; QVector<int> path; };

    QSqlDatabase db;
    QString m_path;
    QHash<QString,int> m_wordId;
    QVector<QString> m_words;
    QVector<double> m_freq, m_ends;
    std::mt19937 m_rng{ std::random_device{}() };
    QVector<int> m_lastPath;
    QString m_lastAgent;
    int m_newEntities = 0, m_lastSentCount = 0, m_lastTokCount = 0;

    // ---------------- random helpers ----------------
    double uniform01() { std::uniform_real_distribution<double> d(0.0, 1.0); return d(m_rng); }
    int uniformInt(int n) { if (n <= 0) return 0;
        std::uniform_int_distribution<int> d(0, n - 1); return d(m_rng); }
    QString pickFrom(const QStringList &l) { return l.at(uniformInt(l.size())); }

    int weightedPick(const QVector<QPair<int,double>> &v)
    {
        if (v.isEmpty()) return -1;
        double total = 0;
        for (const auto &p : v) total += std::max(0.0, p.second);
        if (total <= 1e-12) return v.at(uniformInt(v.size())).first;
        double r = uniform01() * total, acc = 0;
        for (const auto &p : v) { acc += std::max(0.0, p.second); if (r <= acc) return p.first; }
        return v.last().first;
    }

    // ---------------- cache ----------------
    void loadCache()
    {
        m_wordId.clear(); m_words.clear(); m_freq.clear(); m_ends.clear();
        QSqlQuery q(db);
        if (!q.exec("SELECT id,word,freq,ends FROM tokens")) return;
        while (q.next()) {
            int id = q.value(0).toInt();
            while (m_words.size() <= id) { m_words.append(QString()); m_freq.append(0); m_ends.append(0); }
            m_words[id] = q.value(1).toString();
            m_freq[id]  = q.value(2).toDouble();
            m_ends[id]  = q.value(3).toDouble();
            m_wordId.insert(m_words[id], id);
        }
    }
    QString wordOf(int id) const { return (id >= 0 && id < m_words.size()) ? m_words[id] : QString(); }
    double freqOf(int id) const  { return (id >= 0 && id < m_freq.size())  ? m_freq[id] : 0.0; }
    double endsOf(int id) const  { return (id >= 0 && id < m_ends.size())  ? m_ends[id] : 0.0; }

    int tokenId(const QString &w)
    {
        QHash<QString,int>::ConstIterator it = m_wordId.find(w);
        if (it != m_wordId.end()) return it.value();
        QSqlQuery q(db);
        q.prepare("INSERT OR IGNORE INTO tokens(word,freq,ends) VALUES(?,0,0)");
        q.addBindValue(w); q.exec();
        q.prepare("SELECT id FROM tokens WHERE word=?");
        q.addBindValue(w); q.exec();
        int id = -1;
        if (q.next()) id = q.value(0).toInt();
        if (id >= 0) {
            while (m_words.size() <= id) { m_words.append(QString()); m_freq.append(0); m_ends.append(0); }
            m_words[id] = w;
            m_wordId.insert(w, id);
        }
        return id;
    }

    static quint64 packT(int a, int b, int c)
    { return (quint64)(a + 1) << 42 | (quint64)(b + 1) << 21 | (quint64)(c + 1); }
    static void unpackT(quint64 k, int &a, int &b, int &c)
    {
        const quint64 M = 0x1FFFFFull;
        c = int(k & M) - 1; b = int((k >> 21) & M) - 1; a = int(k >> 42) - 1;
    }

    // ---------------- core learning ----------------
    void learnTextCore(const QString &text, const QString &source, int moodHint,
                       int maxSent, bool extractEntities, bool recents)
    {
        QVector<QVector<Tok>> sents = splitSentences(text);
        if (sents.isEmpty()) return;
        db.transaction();
        QSqlQuery qTok(db), qEnd(db), qUp(db), qIns(db), qTmUp(db), qTmIns(db), qSentIns(db);
        qTok.prepare("UPDATE tokens SET freq=freq+? WHERE id=?");
        qEnd.prepare("UPDATE tokens SET ends=ends+1 WHERE id=?");
        qUp.prepare("UPDATE trans SET w=w+? WHERE a=? AND b=? AND c=?");
        qIns.prepare("INSERT INTO trans(a,b,c,w) VALUES(?,?,?,?)");
        qTmUp.prepare("UPDATE tmoods SET w=MIN(w+?,6.0) WHERE token=? AND mood=?");
        qTmIns.prepare("INSERT OR IGNORE INTO tmoods(token,mood,w) VALUES(?,?,?)");
        qSentIns.prepare("INSERT INTO sentences(text,source) VALUES(?,?)");

        int done = 0;
        for (const QVector<Tok> &s : sents) {
            if (done >= maxSent) break;
            ++done;
            if (extractEntities) noteEntitiesInSentence(s, recents);

            QVector<int> ids; ids.reserve(s.size());
            QHash<int,int> delta;
            QStringList origWords;
            QStringList lowers;
            for (const Tok &t : s) {
                int id = tokenId(t.lower);
                if (id < 0) continue;
                ids << id; delta[id]++; origWords << t.orig; lowers << t.lower;
            }
            if (ids.size() < 2) continue;

            for (auto it = delta.constBegin(); it != delta.constEnd(); ++it) {
                qTok.bindValue(0, it.value()); qTok.bindValue(1, it.key()); qTok.exec();
                m_freq[it.key()] += it.value();
            }
            int lastId = ids.last();
            qEnd.bindValue(0, lastId); qEnd.exec();
            m_ends[lastId] += 1;

            // hybrid chain: bigrams (a=-1) + trigrams
            QHash<quint64,double> tr;
            for (int i = 1; i < ids.size(); ++i)
                tr[packT(-1, ids[i-1], ids[i])] += 1.0;
            for (int i = 2; i < ids.size(); ++i)
                tr[packT(ids[i-2], ids[i-1], ids[i])] += 1.0;
            for (auto it = tr.constBegin(); it != tr.constEnd(); ++it) {
                int a, b, c; unpackT(it.key(), a, b, c);
                qUp.bindValue(0, it.value()); qUp.bindValue(1, a);
                qUp.bindValue(2, b); qUp.bindValue(3, c); qUp.exec();
                if (qUp.numRowsAffected() == 0) {
                    qIns.bindValue(0, a); qIns.bindValue(1, b);
                    qIns.bindValue(2, c); qIns.bindValue(3, it.value()); qIns.exec();
                }
            }

            qSentIns.bindValue(0, origWords.join(' ')); qSentIns.bindValue(1, source);
            qSentIns.exec();

            // mood tagging: words co-occurring with mood vocabulary absorb the mood
            int m = moodHint;
            if (m == MoodNone) m = detectMood(lowers);
            if (m != MoodNone) {
                QSet<int> tagged;
                for (int i = 0; i < ids.size(); ++i) {
                    const QString &w = lowers[i];
                    if (w.length() < 3 || stopWords().contains(w) || tagged.contains(ids[i])) continue;
                    tagged.insert(ids[i]);
                    qTmUp.bindValue(0, 0.12); qTmUp.bindValue(1, ids[i]); qTmUp.bindValue(2, m);
                    qTmUp.exec();
                    if (qTmUp.numRowsAffected() == 0) {
                        qTmIns.bindValue(0, ids[i]); qTmIns.bindValue(1, m); qTmIns.bindValue(2, 0.12);
                        qTmIns.exec();
                    }
                }
            }
            m_lastSentCount++; m_lastTokCount += ids.size();
        }
        db.commit();
    }

    int detectMood(const QStringList &lowers) const
    {
        int counts[8] = {0,0,0,0,0,0,0,0};
        const QHash<QString,int> &lex = moodLex();
        for (const QString &w : lowers) {
            QHash<QString,int>::ConstIterator it = lex.find(w);
            if (it != lex.end()) counts[it.value()]++;
        }
        int best = MoodNone, bestN = 0;
        for (int m = 1; m <= 6; ++m) if (counts[m] > bestN) { bestN = counts[m]; best = m; }
        return best;
    }

    bool startsWithQuestionWord(const QStringList &lowers) const
    {
        static const QSet<QString> qw = { "what","why","how","who","when","where","do",
            "does","can","could","tell","is","are","was" };
        return !lowers.isEmpty() && qw.contains(lowers.first());
    }

    // ---------------- entities & pronouns ----------------
    void upsertEntity(const QString &name)
    {
        QSqlQuery q(db);
        q.prepare("UPDATE entities SET mentions=mentions+1 WHERE name=?");
        q.addBindValue(name);
        if (q.exec() && q.numRowsAffected() > 0) return;
        q.prepare("INSERT OR IGNORE INTO entities(name,mentions) VALUES(?,1)");
        q.addBindValue(name); q.exec();
        m_newEntities++;
    }

    void noteEntitiesInSentence(const QVector<Tok> &toks, bool addRecents)
    {
        for (int i = 0; i < toks.size();) {
            if (i > 0 && toks[i].cap) {
                int j = i; QStringList parts;
                while (j < toks.size() && toks[j].cap) { parts << toks[j].orig; ++j; }
                QString name = parts.join(' ').trimmed();
                if (name.length() >= 2 && name.toLower() != "i") {
                    upsertEntity(name);
                    if (addRecents) pushRecent(name);
                }
                i = j;
            } else ++i;
        }
    }

    void pushRecent(const QString &name)
    {
        for (int i = 0; i < recentEntities.size(); ++i)
            if (recentEntities[i].compare(name, Qt::CaseInsensitive) == 0) {
                recentEntities.removeAt(i); break;
            }
        recentEntities.prepend(name);
        while (recentEntities.size() > 12) recentEntities.removeLast();
    }

    QStringList resolvePronouns(const QStringList &lowers, QStringList &notes)
    {
        static const QSet<QString> pron = { "it","they","them","he","she","him","her",
            "that","this","those","these","its","their","his" };
        static const QSet<QString> dem = { "that","this","those","these" };
        QStringList out;
        for (int i = 0; i < lowers.size(); ++i) {
            const QString &w = lowers[i];
            bool hit = pron.contains(w);
            if (hit && dem.contains(w) && i + 1 < lowers.size() &&
                lowers[i+1].length() >= 4 && !stopWords().contains(lowers[i+1]))
                hit = false;                       // "this book" -> keep as determiner
            if (hit && !recentEntities.isEmpty()) {
                QString ent = recentEntities.first();
                out.append(ent.toLower().split(' ', QString::SkipEmptyParts));
                notes.append(QString("%1 → %2").arg(w, ent));
            } else out.append(w);
        }
        return out;
    }

    void persistRecents()
    {
        QJsonArray arr;
        for (const QString &s : recentEntities) arr.append(s);
        kvSet("recent_entities",
              QString::fromUtf8(QJsonDocument(arr).toJson(QJsonDocument::Compact)));
    }
    void loadRecents()
    {
        recentEntities.clear();
        QJsonDocument d = QJsonDocument::fromJson(kvGet("recent_entities").toUtf8());
        if (d.isArray()) for (const QJsonValue &v : d.array()) recentEntities << v.toString();
    }

    // ---------------- generation ----------------
    double moodAff(QSqlQuery &q, int token, int mood)
    {
        if (mood == MoodNone || token < 0) return 0;
        q.bindValue(0, token); q.bindValue(1, mood);
        double w = 0;
        if (q.exec() && q.next()) w = q.value(0).toDouble();
        return qMin(w, 4.0);
    }

    QVector<QPair<int,double>> transitionsAfter(QSqlQuery &q, int a, int b)
    {
        QVector<QPair<int,double>> out;
        q.bindValue(0, a); q.bindValue(1, b);
        if (q.exec()) while (q.next())
            out.append({ q.value(0).toInt(), q.value(1).toDouble() });
        return out;
    }

    int sampleUnigram(int mood, bool rareBias)
    {
        if (m_words.isEmpty()) return -1;
        QSqlQuery q(db), qm(db);
        q.prepare("SELECT id,freq FROM tokens ORDER BY RANDOM() LIMIT 90");
        qm.prepare("SELECT w FROM tmoods WHERE token=? AND mood=?");
        QVector<QPair<int,double>> v;
        if (q.exec()) while (q.next()) {
            int id = q.value(0).toInt();
            double f = q.value(1).toDouble();
            double w = rareBias ? 1.0 / (f + 1.0) : f;
            w *= (1.0 + 2.0 * moodAff(qm, id, mood));
            v.append({ id, w });
        }
        return weightedPick(v);
    }

    // the hybrid walk: trigram -> bigram -> unigram, mood-weighted sampling
    QVector<int> walkChain(int startId, int mood, double temp, int maxLen,
                           double moodBoost, bool rareBias)
    {
        QVector<int> path;
        if (startId < 0) startId = sampleUnigram(mood, rareBias);
        if (startId < 0) return path;
        path.append(startId);
        QSqlQuery qTri(db), qMood(db);
        qTri.prepare("SELECT c,w FROM trans WHERE a=? AND b=?");
        qMood.prepare("SELECT w FROM tmoods WHERE token=? AND mood=?");
        for (int step = 0; step < maxLen; ++step) {
            int prev1 = path.last();
            int prev2 = path.size() >= 2 ? path[path.size()-2] : -2;
            QVector<QPair<int,double>> cand;
            if (prev2 >= 0) cand = transitionsAfter(qTri, prev2, prev1);
            if (cand.isEmpty() || uniform01() < 0.15)
                cand = transitionsAfter(qTri, -1, prev1);
            if (cand.isEmpty()) {
                int u = sampleUnigram(mood, rareBias);
                if (u < 0) break;
                cand.append({ u, 1.0 });
            }
            for (auto &p : cand) {
                double w = rareBias ? 1.0 / (freqOf(p.first) + 1.0) : p.second;
                w *= (1.0 + moodBoost * moodAff(qMood, p.first, mood));
                p.second = std::pow(std::max(w, 1e-6), 1.0 / std::max(temp, 0.05));
            }
            int next = weightedPick(cand);
            if (next < 0) break;
            path.append(next);
            if (next == startId && path.size() > 8) break;
            if (path.size() >= 6) {
                double endP = endsOf(next) / std::max(1.0, freqOf(next));
                if (uniform01() < endP) break;
            }
        }
        return path;
    }

    QString detokenize(const QVector<int> &path, bool capitalize = true) const
    {
        QStringList ws;
        for (int id : path) ws << wordOf(id);
        QString s = ws.join(' ');
        if (capitalize && !s.isEmpty()) s[0] = s.at(0).toUpper();
        return s + (s.endsWith('.') ? QString() : QStringLiteral("."));
    }

    int pickRareSeed(const QVector<int> &ids)
    {
        QVector<QPair<int,double>> v;
        QSet<int> seen;
        for (int id : ids) {
            if (id < 0 || seen.contains(id)) continue;
            seen.insert(id);
            v.append({ id, 1.0 / (freqOf(id) + 1.0) });
        }
        return weightedPick(v);
    }

    QStringList keywordsOf(const QStringList &lowers) const
    {
        QSet<QString> seen; QStringList out;
        for (const QString &w : lowers) {
            if (w.length() < 4 || stopWords().contains(w) || seen.contains(w)) continue;
            seen.insert(w); out.append(w);
        }
        std::sort(out.begin(), out.end(), [&](const QString &a, const QString &b) {
            auto ita = m_wordId.find(a); auto itb = m_wordId.find(b);
            double fa = (ita != m_wordId.end()) ? m_freq[ita.value()] : 1e9;
            double fb = (itb != m_wordId.end()) ? m_freq[itb.value()] : 1e9;
            return fa < fb;                                   // rarest first
        });
        return out;
    }

    double agentBoost(const QString &agent)
    { return kvGet("boost_" + agent, "1.0").toDouble(); }

    // ---------------- the four agents ----------------
    Cand librarianReply(const QStringList &resolved)   // quotes real sentences
    {
        Cand c; c.agent = "Librarian";
        QStringList kws = keywordsOf(resolved);
        struct Hit { int n = 0; QString text, source; };
        QMap<int, Hit> hits;
        int searched = 0;
        for (const QString &kw : kws) {
            if (searched++ >= 6) break;
            QSqlQuery q(db);
            q.prepare("SELECT id,text,source FROM sentences "
                      "WHERE text LIKE ? AND source != 'conversation' LIMIT 120");
            q.addBindValue("%" + kw + "%");
            if (!q.exec()) continue;
            while (q.next()) {
                int id = q.value(0).toInt();
                Hit &h = hits[id]; h.n++;
                if (h.text.isEmpty()) { h.text = q.value(1).toString(); h.source = q.value(2).toString(); }
            }
        }
        QString curSrc = kvGet("current_source");
        if (hits.isEmpty() && !curSrc.isEmpty()) {     // "what is it about?" style
            QSqlQuery q(db);
            q.prepare("SELECT text,source FROM sentences WHERE source=? ORDER BY RANDOM() LIMIT 1");
            q.addBindValue(curSrc);
            if (q.exec() && q.next()) {
                c.text = QStringLiteral("“%1” — from %2.").arg(q.value(0).toString(), curSrc);
                c.score = 0.55;
                return c;
            }
        }
        if (hits.isEmpty()) return c;
        QVector<QPair<int,double>> pool;
        for (auto it = hits.constBegin(); it != hits.constEnd(); ++it) {
            double s = it.value().n;
            for (const QString &kw : kws)
                if (it.value().source.contains(kw, Qt::CaseInsensitive)) s += 1.5;
            s += uniform01() * 0.4;
            pool.append({ it.key(), s });
        }
        int chosen = weightedPick(pool);
        Hit h = hits.value(chosen);
        c.text = QStringLiteral("“%1”").arg(h.text);
        if (!h.source.isEmpty() && uniform01() < 0.6)
            c.text += QStringLiteral(" — I remember that from %1.").arg(h.source);
        c.score = 0.9 + qMin(2.0, h.n * 0.4);
        return c;
    }

    Cand analystReply(const QVector<int> &seedIds, int mood, bool isAck)
    {
        Cand c; c.agent = "Analyst";
        int seed = -1;
        if (isAck && !m_lastPath.isEmpty()) seed = m_lastPath.last();  // continue thread
        else if (!seedIds.isEmpty()) seed = pickRareSeed(seedIds);
        QVector<int> path = walkChain(seed, mood, temperature * 0.8, 22, 0.9, false);
        if (path.size() < 3) return c;
        c.path = path;
        QString pre;
        if (uniform01() < 0.35)
            pre = pickFrom({ "From what I have read, ", "Tracing the chain, ",
                             "As I understand it, ", "The pattern suggests that " });
        c.text = pre.isEmpty() ? detokenize(path) : pre + detokenize(path, false);
        c.score = 0.85 + 0.03 * path.size();
        return c;
    }

    Cand echoReply(const QVector<int> &seedIds, int mood, int userMood)
    {
        Cand c; c.agent = "Echo";
        int moodUse = mood;
        double boost = 2.2;
        if (moodUse == MoodNone) {
            moodUse = kvGet("last_mood", "0").toInt();
            boost = 1.0;
            if (moodUse == MoodNone) moodUse = MoodCalm;
        }
        int seed = !seedIds.isEmpty() ? pickRareSeed(seedIds) : -1;
        QVector<int> path = walkChain(seed, moodUse, temperature * 1.05, 20, boost, false);
        if (path.size() < 3) return c;
        c.path = path;
        QString pre;
        double p = (userMood != MoodNone) ? 0.65 : 0.25;
        if (uniform01() < p) {
            switch (moodUse) {
            case MoodJoy:       pre = "That feels bright to me — "; break;
            case MoodSorrow:    pre = "I hear something heavy in that — "; break;
            case MoodCuriosity: pre = "Ooh, a thread worth chasing — "; break;
            case MoodFear:      pre = "Careful, that sounds unsettling — "; break;
            case MoodAnger:     pre = "Steam detected — "; break;
            default:            pre = "Let's keep this gentle — "; break;
            }
        }
        c.text = pre.isEmpty() ? detokenize(path) : pre + detokenize(path, false);
        c.score = 0.8 + (userMood != MoodNone ? 0.5 : 0.0) + (forcedMood != MoodNone ? 0.3 : 0.0);
        return c;
    }

    Cand dreamerReply(int mood)
    {
        Cand c; c.agent = "Dreamer";
        int seed = sampleUnigram(mood, true);
        QVector<int> path = walkChain(seed, mood, temperature * 1.6, 24, 0.4, true);
        if (path.size() < 3) return c;
        c.path = path;
        QString pre;
        if (uniform01() < 0.4)
            pre = pickFrom({ "Oddly, my circuits drift — ", "In a parallel dream, ",
                             "Somewhere in the noise, " });
        c.text = pre.isEmpty() ? detokenize(path) : pre + detokenize(path, false);
        c.score = 0.7;
        return c;
    }

    static QString seedCorpus()
    {
        return QStringLiteral(
            "Hello, I am a tiny mind made of Markov chains and memories.\n"
            "Ada kept a notebook of every word she ever heard.\n"
            "The lighthouse keeper watched the dark sea and waited for ships.\n"
            "When the storm arrived, the keeper felt fear but stayed brave.\n"
            "Ada loved the smell of old paper and bright morning light.\n"
            "Curiosity is a quiet engine that asks questions all day long.\n"
            "The old library held a thousand sleeping stories on its shelves.\n"
            "Sometimes the machines dream in probabilities and wake up smiling.\n"
            "A good memory is a garden where every sentence is a seed.\n"
            "The fox crossed the silver field under a cold winter moon.\n"
            "Sad songs drift through empty halls like slow grey rain.\n"
            "Wonder is the spark that turns a question into a journey.\n"
            "The captain trusted the compass more than the weather.\n"
            "Every word you speak teaches me a new path to walk.\n"
            "Joy bubbles up when two minds find the same idea.\n"
            "The clock on the wall measured silence as carefully as time.\n");
    }
};

// ---------------------------------------------------------------------------
// GUI
// ---------------------------------------------------------------------------
class MainWindow : public QMainWindow
{
public:
    MainWindow(Brain *b, const QString &dbPath) : brain(b), m_dbPath(dbPath)
    {
        setWindowTitle("AgentMind — hybrid Markov multi-agent demo");
        resize(1180, 740);

        QSplitter *split = new QSplitter(Qt::Horizontal, this);

        // ---------- left panel ----------
        QWidget *left = new QWidget;
        QVBoxLayout *lv = new QVBoxLayout(left);
        lv->setContentsMargins(6, 6, 6, 6);
        QTabWidget *tabs = new QTabWidget;

        // Agents tab
        QWidget *agentsTab = new QWidget;
        QVBoxLayout *av = new QVBoxLayout(agentsTab);
        av->addWidget(new QLabel("<b>Agents</b> (uncheck to mute)"));
        agentList = new QListWidget;
        struct AI { const char *name, *desc; } infos[4] = {
            { "Librarian", "Quotes real sentences from the files it read. Strongest on questions." },
            { "Analyst",   "Low-temperature Markov walk. Explains; continues the thread on short replies." },
            { "Echo",      "Mood mirror. Amplifies detected/forced mood in word choices." },
            { "Dreamer",   "High-temperature rare-word walker. Surreal jumps." },
        };
        for (const AI &a : infos) {
            QListWidgetItem *it = new QListWidgetItem(
                QString("%1 — %2").arg(a.name, a.desc));
            it->setData(Qt::UserRole, a.name);
            it->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsUserCheckable);
            it->setCheckState(Qt::Checked);
            agentList->addItem(it);
        }
        av->addWidget(agentList, 1);
        directorLabel = new QLabel("Last speaker: —");
        av->addWidget(directorLabel);
        tabs->addTab(agentsTab, "Agents");

        // Memory tab
        QWidget *memTab = new QWidget;
        QVBoxLayout *mv = new QVBoxLayout(memTab);
        mv->addWidget(new QLabel("<b>Recent entities</b> (pronouns resolve here)"));
        recentLabel = new QLabel("—"); recentLabel->setWordWrap(true);
        mv->addWidget(recentLabel);
        mv->addWidget(new QLabel("<b>Top entities</b>"));
        entityList = new QListWidget;
        mv->addWidget(entityList, 1);
        mv->addWidget(new QLabel("<b>Files read</b>"));
        sourcesLabel = new QLabel("—"); sourcesLabel->setWordWrap(true);
        mv->addWidget(sourcesLabel);
        QHBoxLayout *mb = new QHBoxLayout;
        QPushButton *forgetBtn = new QPushButton("Forget chat log");
        QPushButton *wipeBtn = new QPushButton("Full wipe");
        mb->addWidget(forgetBtn); mb->addWidget(wipeBtn);
        mv->addLayout(mb);
        tabs->addTab(memTab, "Memory");

        // Brain tab
        QWidget *brainTab = new QWidget;
        QVBoxLayout *bv = new QVBoxLayout(brainTab);
        statsLabel = new QLabel; statsLabel->setWordWrap(true);
        QFont mono("Monospace"); mono.setStyleHint(QFont::TypeWriter);
        statsLabel->setFont(mono);
        bv->addWidget(statsLabel);
        QLabel *tempTitle = new QLabel("<b>Global temperature</b> (higher = wilder)");
        tempSlider = new QSlider(Qt::Horizontal);
        tempSlider->setRange(50, 200); tempSlider->setValue(100);
        tempValue = new QLabel("1.00");
        QHBoxLayout *tr = new QHBoxLayout;
        tr->addWidget(tempSlider, 1); tr->addWidget(tempValue);
        bv->addWidget(tempTitle); bv->addLayout(tr);
        bv->addStretch(1);
        tabs->addTab(brainTab, "Brain");

        lv->addWidget(tabs);
        split->addWidget(left);

        // ---------- right panel: chat ----------
        QWidget *right = new QWidget;
        QVBoxLayout *rv = new QVBoxLayout(right);
        rv->setContentsMargins(6, 6, 6, 6);
        chat = new QTextBrowser;
        chat->setOpenLinks(false);
        rv->addWidget(chat, 1);
        QHBoxLayout *row = new QHBoxLayout;
        input = new QLineEdit;
        input->setPlaceholderText(
            "Chat… try: “what is it about?”, “tell me about Ada”, “continue”, “yes”");
        QPushButton *send = new QPushButton("Send");
        btnUp = new QPushButton(QStringLiteral("👍"));
        btnDown = new QPushButton(QStringLiteral("👎"));
        btnUp->setToolTip("Reinforce the transitions used in the last answer");
        btnDown->setToolTip("Decay the transitions used in the last answer");
        row->addWidget(input, 1);
        row->addWidget(send);
        row->addWidget(btnUp);
        row->addWidget(btnDown);
        rv->addLayout(row);
        split->addWidget(right);

        split->setStretchFactor(0, 0);
        split->setStretchFactor(1, 1);
        split->setSizes({ 330, 850 });
        setCentralWidget(split);

        // ---------- toolbar ----------
        QToolBar *tb = addToolBar("Main");
        QAction *openAct = tb->addAction(QStringLiteral("📚 Read files…"));
        tb->addWidget(new QLabel("  Mood: "));
        moodBox = new QComboBox;
        moodBox->addItems({ "Auto-detect", "Joyful", "Melancholic", "Curious",
                            "Fearful", "Angry", "Calm" });
        tb->addWidget(moodBox);
        tb->addSeparator();
        QAction *wipeAct = tb->addAction(QStringLiteral("🧹 Wipe memory"));

        statusBar()->showMessage("starting…");

        // ---------- connections ----------
        connect(openAct, &QAction::triggered, this, &MainWindow::openFiles);
        connect(wipeAct, &QAction::triggered, this, &MainWindow::wipeMemory);
        connect(send, &QPushButton::clicked, this, &MainWindow::send);
        connect(input, &QLineEdit::returnPressed, this, &MainWindow::send);
        connect(btnUp, &QPushButton::clicked, [this] {
            if (brain->feedback(true))
                addSystem(QStringLiteral("👍 Reinforced the pathways of the last answer (agent %1 got a boost).")
                          .arg(brain->lastAgent()));
        });
        connect(btnDown, &QPushButton::clicked, [this] {
            if (brain->feedback(false))
                addSystem(QStringLiteral("👎 Decayed the pathways of the last answer."));
        });
        connect(moodBox, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int idx) {
            brain->forcedMood = idx;              // 0 == auto
        });
        connect(tempSlider, &QSlider::valueChanged, [this](int v) {
            brain->temperature = v / 100.0;
            tempValue->setText(QString::number(v / 100.0, 'f', 2));
        });
        connect(agentList, &QListWidget::itemChanged, [this](QListWidgetItem *) {
            brain->mutedAgents.clear();
            for (int i = 0; i < agentList->count(); ++i) {
                QListWidgetItem *x = agentList->item(i);
                if (x->checkState() != Qt::Checked)
                    brain->mutedAgents.insert(x->data(Qt::UserRole).toString());
            }
        });
        connect(forgetBtn, &QPushButton::clicked, [this] {
            brain->forgetConversation();
            chat->clear();
            addSystem("Chat log cleared. Everything I read is still inside me.");
            refreshSide();
        });

        // ---------- style ----------
        qApp->setStyleSheet(
            "QMainWindow, QWidget { background:#1e242b; color:#dfe6ee; font-size:13px; }"
            "QTextBrowser { background:#171c22; border:1px solid #2c3540; }"
            "QLineEdit { background:#232b34; border:1px solid #39434e; border-radius:6px; padding:7px; }"
            "QPushButton { background:#2f6fb2; color:white; border:none; border-radius:6px; padding:7px 14px; }"
            "QPushButton:hover { background:#3d84cc; }"
            "QComboBox { background:#232b34; border:1px solid #39434e; border-radius:4px; padding:3px 8px; }"
            "QComboBox QAbstractItemView { background:#232b34; color:#dfe6ee; selection-background-color:#2f6fb2; }"
            "QListWidget { background:#20262e; border:1px solid #2c3540; }"
            "QTabWidget::pane { border:1px solid #2c3540; }"
            "QTabBar::tab { background:#232b34; padding:6px 12px; }"
            "QTabBar::tab:selected { background:#2f6fb2; color:white; }"
            "QStatusBar { background:#161b21; }"
            "QToolBar { background:#1a2027; border-bottom:1px solid #2c3540; spacing:6px; padding:4px; }");

        // ---------- restore chat history ----------
        QVector<Brain::Episode> eps = brain->episodes();
        if (eps.isEmpty()) {
            showIntro();
        } else {
            addSystem("Memory restored from SQLite — resuming conversation.");
            for (const Brain::Episode &e : eps) {
                if (e.role == "user") addUser(e.content);
                else {
                    int p = e.meta.indexOf('|');
                    QString agent = (p >= 0) ? e.meta.left(p) : "Bot";
                    QString notes = (p >= 0) ? e.meta.mid(p + 1) : "";
                    addBot(e.content, agent.isEmpty() ? "Bot" : agent, notes);
                }
            }
        }
        refreshSide();
        input->setFocus();
    }

private:
    Brain *brain;
    QString m_dbPath;
    QTextBrowser *chat;
    QLineEdit *input;
    QComboBox *moodBox;
    QListWidget *agentList, *entityList;
    QLabel *directorLabel, *recentLabel, *sourcesLabel, *statsLabel, *tempValue;
    QSlider *tempSlider;
    QPushButton *btnUp, *btnDown;

    static QString agentColor(const QString &a)
    {
        if (a == "Librarian") return "#b48ce8";
        if (a == "Analyst")   return "#6fb3ff";
        if (a == "Echo")      return "#7fd6a2";
        if (a == "Dreamer")   return "#ffb36b";
        return "#dfe6ee";
    }

    void ensureScroll()
    {
        QTextCursor c = chat->textCursor();
        c.movePosition(QTextCursor::End);
        chat->setTextCursor(c);
    }

    void addSystem(const QString &s)
    {
        chat->append(QStringLiteral("<p style=\"color:#8a97a6;\"><i>%1</i></p>").arg(htmlEscape(s)));
        ensureScroll();
    }
    void addUser(const QString &s)
    {
        chat->append(QStringLiteral(
            "<table width=\"100%\" cellpadding=\"8\" cellspacing=\"0\"><tr>"
            "<td width=\"26%\"></td>"
            "<td bgcolor=\"#1f3a56\"><b style=\"color:#7db8ff;\">You ›</b>&nbsp; %1</td>"
            "</tr></table>").arg(htmlEscape(s)));
        ensureScroll();
    }
    void addBot(const QString &s, const QString &agent, const QString &notes)
    {
        QString html = QStringLiteral(
            "<table width=\"100%\" cellpadding=\"8\" cellspacing=\"0\"><tr>"
            "<td bgcolor=\"#26313d\"><b style=\"color:%1;\">%2 ›</b>&nbsp; %3</td>"
            "<td width=\"26%\"></td></tr></table>")
            .arg(agentColor(agent), htmlEscape(agent), htmlEscape(s));
        if (!notes.isEmpty())
            html += QStringLiteral("<p style=\"color:#778392; font-size:small;\">&nbsp;&nbsp;🧭 %1</p>")
                        .arg(htmlEscape(notes));
        chat->append(html);
        ensureScroll();
    }

    void showIntro()
    {
        addSystem("Welcome! This is a hybrid multi-agent Markov brain with SQLite long-term memory.");
        addSystem("📚 Read files… teaches it · the Mood menu steers tone · pronouns like 'it/they/that' "
                  "resolve to recent entities · 👍/👎 reinforce or decay its last pathways.");
        addBot("Hello! I am a small swarm of four agents sharing one Markov memory. "
               "Read me some text, then ask me about it — you can use pronouns like 'it' or 'they', "
               "and set my mood from the toolbar.",
               "Director", "first boot");
    }

    void send()
    {
        QString text = input->text().trimmed();
        if (text.isEmpty()) return;
        input->clear();
        addUser(text);
        Brain::Reply r = brain->respond(text);
        QString notes = QStringLiteral("mood: %1").arg(r.moodName);
        if (!r.notes.isEmpty()) notes += " · " + r.notes.join(" · ");
        addBot(r.text, r.agent, notes);
        directorLabel->setText("Last speaker: " + r.agent);
        refreshSide();
    }

    void openFiles()
    {
        QStringList files = QFileDialog::getOpenFileNames(
            this, "Read text files", QString(),
            "Text files (*.txt *.md *.text);;All files (*)");
        if (files.isEmpty()) return;
        QApplication::setOverrideCursor(Qt::WaitCursor);
        for (const QString &f : files) {
            QFile file(f);
            if (!file.open(QIODevice::ReadOnly)) {
                addSystem("Could not open " + f);
                continue;
            }
            QString text = QString::fromUtf8(file.readAll());
            QFileInfo fi(f);
            Brain::LearnStats st = brain->learnFile(text, fi.fileName());
            addSystem(QStringLiteral("📖 Read '%1': %2 sentences · %3 tokens · %4 entities in %5 ms. "
                                     "Ask me about it!")
                          .arg(fi.fileName()).arg(st.sentences).arg(st.tokens)
                          .arg(st.entities).arg(st.ms));
        }
        QApplication::restoreOverrideCursor();
        refreshSide();
    }

    void wipeMemory()
    {
        if (QMessageBox::question(this, "Wipe memory",
                "Delete ALL learned knowledge, chat history and entities?") != QMessageBox::Yes)
            return;
        brain->wipe();
        chat->clear();
        showIntro();
        refreshSide();
    }

    void refreshSide()
    {
        Brain::Stats st = brain->stats();
        statsLabel->setText(QStringLiteral(
            "tokens        %1\ntransitions   %2\nsentences     %3\nepisodes      %4\n"
            "entities      %5\npraise        %6\ncriticism     %7")
            .arg(st.tokens).arg(st.transitions).arg(st.sentences).arg(st.episodes)
            .arg(st.entities)
            .arg(brain->kvGet("praise", "0"), brain->kvGet("criticism", "0")));
        entityList->clear();
        entityList->addItems(brain->topEntities(40));
        recentLabel->setText(brain->recentEntities.isEmpty()
                             ? "—" : brain->recentEntities.join(", "));
        sourcesLabel->setText(brain->kvGet("sources").split('|', QString::SkipEmptyParts).join(", "));
        statusBar()->showMessage(QStringLiteral("memory: %1   ·   %2 tokens · %3 transitions · %4 sentences")
                                     .arg(m_dbPath).arg(st.tokens).arg(st.transitions).arg(st.sentences));
    }
};

// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling, true);
    QApplication app(argc, argv);
    app.setApplicationName("AgentMind");
    app.setOrganizationName("AgentMindDemo");

    if (!QSqlDatabase::isDriverAvailable("QSQLITE")) {
        QMessageBox::critical(nullptr, "AgentMind",
            "QSQLITE driver not found.\nAvailable drivers: " +
            QSqlDatabase::drivers().join(", "));
        return 1;
    }

    QString dir = QApplication::applicationDirPath();
    QDir().mkpath(dir);
    QString dbPath = dir + "/agentmind.sqlite";

    Brain brain;
    QString err;
    if (!brain.open(dbPath, &err)) {
        QMessageBox::critical(nullptr, "AgentMind",
            "Could not open SQLite memory:\n" + err);
        return 1;
    }

    MainWindow w(&brain, dbPath);
    w.show();
    return app.exec();
}
