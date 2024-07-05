#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <sqlite3.h>
#include <algorithm>
#include <ctime>

// Simplified POS Tagging
enum class POSType { NOUN, VERB, ADJECTIVE, UNKNOWN };

POSType get_pos_type(const std::string& word) {
    // Simplified POS tagging based on word endings (not accurate, for demonstration only)
    if (word.size() > 2 && word.substr(word.size() - 2) == "ed") {
        return POSType::VERB;
    }
    if (word.size() > 3 && word.substr(word.size() - 3) == "ing") {
        return POSType::VERB;
    }
    if (word.size() > 3 && word.substr(word.size() - 3) == "ous") {
        return POSType::ADJECTIVE;
    }
    if (word.size() > 4 && word.substr(word.size() - 4) == "ness") {
        return POSType::NOUN;
    }
    return POSType::UNKNOWN;
}

class SimpleBrain {
public:
    SimpleBrain(const std::string& name) : name(name) {
        open_database();
        create_dictionary_table();
        load_stop_words("stop_words.txt");  // Load stop words from a file
    }

    ~SimpleBrain() {
        close_database();
    }

    void load_dictionary(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Error opening dictionary file: " + filename);
        }

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string word, rest;
            std::getline(iss, word, ':');
            std::getline(iss, rest);

            std::string definition, example1, example2;
            std::istringstream restStream(rest);
            std::getline(restStream, definition, '|');
            std::getline(restStream, example1, '|');
            std::getline(restStream, example2, '|');

            add_word_to_dictionary(word, definition, example1, example2);
        }
        std::cout << "Dictionary loaded successfully." << std::endl;
    }

    std::string get_word_definition(const std::string& word) {
        std::string sql = "SELECT definition FROM dictionary WHERE word = ?";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
            std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
            return "";
        }

        if (sqlite3_bind_text(stmt, 1, word.c_str(), -1, SQLITE_STATIC) != SQLITE_OK) {
            std::cerr << "Failed to bind parameter: " << sqlite3_errmsg(db) << std::endl;
            sqlite3_finalize(stmt);
            return "";
        }

        std::string definition;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            definition = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        } else {
            definition = "Definition not found.";
        }

        sqlite3_finalize(stmt);
        return definition;
    }

    std::vector<std::string> get_word_examples(const std::string& word) {
        std::string sql = "SELECT example1, example2 FROM dictionary WHERE word = ?";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
            std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
            return {};
        }

        if (sqlite3_bind_text(stmt, 1, word.c_str(), -1, SQLITE_STATIC) != SQLITE_OK) {
            std::cerr << "Failed to bind parameter: " << sqlite3_errmsg(db) << std::endl;
            sqlite3_finalize(stmt);
            return {};
        }

        std::vector<std::string> examples;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            examples.push_back(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
            examples.push_back(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
        }

        sqlite3_finalize(stmt);
        return examples;
    }

    // Conversation memory
    void add_to_memory(const std::string& user_input, const std::string& bot_response) {
        conversation_memory.push_back({user_input, bot_response});
        update_word_frequencies(user_input);
        if (conversation_memory.size() > memory_limit) {
            conversation_memory.erase(conversation_memory.begin());
        }
    }

    // Generate response using word frequency and POS analysis
    std::string generate_response(const std::string& input) {
        std::vector<std::string> words = tokenize(input);

        if (words.empty()) {
            return "I don't understand.";
        }

        // Find the most important word (not a stop word) in the input
        std::string important_word = find_most_important_word(words);
        if (important_word.empty()) {
            return "I don't have enough information.";
        }

        // Generate a response based on the most important word
        std::string response = "Let's talk more about " + important_word + ". ";
        response += generate_sentence_from_word(important_word);

        return response;
    }

    // Rate the response and store it
    void rate_response(const std::string& user_input, const std::string& bot_response, int rating) {
        rated_responses.push_back({user_input, bot_response, rating});
    }

    // Generate a response from historical rated responses
    std::string generate_response_from_history(const std::string& input) {
        std::vector<std::string> words = tokenize(input);
        if (words.empty()) {
            return "I don't understand.";
        }

        // Find the most important word
        std::string important_word = find_most_important_word(words);
        if (important_word.empty()) {
            return "I don't have enough information.";
        }

        // Group responses by rating
        std::unordered_map<int, std::vector<std::string>> responses_by_rating;
        for (const auto& [user_input, response, rating] : rated_responses) {
            if (user_input.find(important_word) != std::string::npos) {
                responses_by_rating[rating].push_back(response);
            }
        }

        // Choose a response from the highest-rated group
        if (!responses_by_rating.empty()) {
            int max_rating = std::max_element(responses_by_rating.begin(), responses_by_rating.end(),
                                              [](const auto& a, const auto& b) { return a.first < b.first; })
                             ->first;
            const auto& best_responses = responses_by_rating[max_rating];

            // Choose a random response from the best group
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, best_responses.size() - 1);

            return best_responses[dis(gen)];
        }

        return generate_response(input);  // Fallback to normal response generation
    }

    std::string get_name() const {
        return name;
    }

private:
    std::string name;
    sqlite3* db;
    std::vector<std::pair<std::string, std::string>> conversation_memory;
    std::vector<std::tuple<std::string, std::string, int>> rated_responses;
    std::unordered_map<std::string, int> word_frequencies;
    std::unordered_set<std::string> stop_words;
    const size_t memory_limit = 100;
    const size_t response_length = 10;

    void open_database() {
        int rc = sqlite3_open("word_context.db", &db);
        if (rc) {
            std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
            return;
        }
    }

    void close_database() {
        sqlite3_close(db);
    }

    void create_dictionary_table() {
        const char* sql = "CREATE TABLE IF NOT EXISTS dictionary (word TEXT PRIMARY KEY, definition TEXT, example1 TEXT, example2 TEXT)";
        char* errmsg;
        if (sqlite3_exec(db, sql, nullptr, nullptr, &errmsg) != SQLITE_OK) {
            std::cerr << "Failed to create table: " << errmsg << std::endl;
            sqlite3_free(errmsg);
        }
    }

    void add_word_to_dictionary(const std::string& word, const std::string& definition, const std::string& example1, const std::string& example2) {
        std::string sql = "INSERT OR REPLACE INTO dictionary (word, definition, example1, example2) VALUES (?, ?, ?, ?)";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
            std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
            return;
        }

        if (sqlite3_bind_text(stmt, 1, word.c_str(), -1, SQLITE_STATIC) != SQLITE_OK ||
            sqlite3_bind_text(stmt, 2, definition.c_str(), -1, SQLITE_STATIC) != SQLITE_OK ||
            sqlite3_bind_text(stmt, 3, example1.c_str(), -1, SQLITE_STATIC) != SQLITE_OK ||
            sqlite3_bind_text(stmt, 4, example2.c_str(), -1, SQLITE_STATIC) != SQLITE_OK) {
            std::cerr << "Failed to bind parameters: " << sqlite3_errmsg(db) << std::endl;
            sqlite3_finalize(stmt);
            return;
        }

        if (sqlite3_step(stmt) != SQLITE_DONE) {
            std::cerr << "Failed to execute statement: " << sqlite3_errmsg(db) << std::endl;
        }

        sqlite3_finalize(stmt);
    }

    void update_word_frequencies(const std::string& input) {
        std::istringstream iss(input);
        std::string word;
        while (iss >> word) {
            if (stop_words.find(word) == stop_words.end()) {
                word_frequencies[word]++;
            }
        }
    }

    std::vector<std::string> tokenize(const std::string& text) {
        std::istringstream iss(text);
        std::string word;
        std::vector<std::string> words;
        while (iss >> word) {
            words.push_back(word);
        }
        return words;
    }

    void load_stop_words(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Error opening stop words file: " + filename);
        }

        std::string line;
        while (std::getline(file, line)) {
            stop_words.insert(line);
        }
    }

    std::string find_most_important_word(const std::vector<std::string>& words) {
        std::unordered_map<std::string, int> word_importance;
        for (const auto& word : words) {
            if (stop_words.find(word) == stop_words.end()) {
                word_importance[word] = word_frequencies[word];
            }
        }

        auto max_element = std::max_element(word_importance.begin(), word_importance.end(),
                                            [](const auto& a, const auto& b) { return a.second < b.second; });

        return max_element != word_importance.end() ? max_element->first : "";
    }

    std::string generate_sentence_from_word(const std::string& word) {
        auto examples = get_word_examples(word);
        if (!examples.empty()) {
            return examples[0];
        }
        return "I don't have enough information about " + word + ".";
    }
};

class BrainManager {
public:
    void add_brain(const SimpleBrain& brain) {
        brains.push_back(brain);
    }

    std::string generate_best_response(const std::string& input) {
        std::vector<std::string> responses;
        for (const auto& brain : brains) {
            responses.push_back(brain.generate_response(input));
        }

        // Simple voting mechanism to choose the best response
        std::unordered_map<std::string, int> response_votes;
        for (const auto& response : responses) {
            response_votes[response]++;
        }

        auto best_response = std::max_element(response_votes.begin(), response_votes.end(),
                                              [](const auto& a, const auto& b) { return a.second < b.second; });

        return best_response != response_votes.end() ? best_response->first : "I'm not sure how to respond.";
    }

private:
    std::vector<SimpleBrain> brains;
};

int main() {
    SimpleBrain brain1("Brain1");
    brain1.load_dictionary("dictionary1.txt");

    SimpleBrain brain2("Brain2");
    brain2.load_dictionary("dictionary2.txt");

    BrainManager brain_manager;
    brain_manager.add_brain(brain1);
    brain_manager.add_brain(brain2);

    std::string input;
    while (true) {
        std::cout << "You: ";
        std::getline(std::cin, input);
        if (input == "exit") break;

        std::string response = brain_manager.generate_best_response(input);
        std::cout << "Bot: " << response << std::endl;
    }

    return 0;
}

