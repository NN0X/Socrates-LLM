#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <climits>
#include <thread>
#include <mutex>
#include <cstdint>
#include <queue>

//#define DEBUG
#define VERBOSE

#include "tokenizer.h"

void calculateFrequencyThread(std::unordered_map<std::string, size_t>& frequencyCount,
                              std::unordered_map<std::string, size_t>& threadSpecificFrequencyCount,
                              const std::unordered_map<size_t, std::unordered_set<std::string>>& wordsByLength,
                              std::unordered_map<std::string, size_t>& checkedTokens,
                              std::priority_queue<std::pair<size_t, std::string>>& checkedTokensPriorityQueue,
                              std::mutex& frequencyCountMutex)
{
        for (const auto& token1 : threadSpecificFrequencyCount)
        {
                for (const auto& token2 : frequencyCount)
                {
                        std::string newToken = token1.first + token2.first;
                        if (newToken.size() > MAX_WORD_LENGTH || checkedTokens.find(newToken) != checkedTokens.end())
                        {
                                continue;
                        }
                        size_t frequency = 0;
                        for (const auto& wordsLengthBracket : wordsByLength)
                        {
                                if (wordsLengthBracket.first < newToken.size())
                                {
                                        continue;
                                }
                                for (const std::string& word : wordsLengthBracket.second)
                                {
                                        if (word.find(newToken) != std::string::npos)
                                        {
                                                frequency++;
                                        }
                                }
                        }

                        {
                                std::lock_guard<std::mutex> lock(frequencyCountMutex);
                                checkedTokens[newToken] = frequency;
                                checkedTokensPriorityQueue.push(std::make_pair(frequency, newToken));
                        }
                }
        }
}

void addTokens(const std::unordered_map<size_t, std::unordered_set<std::string>> &wordsByLength, std::unordered_map<std::string, size_t>& frequencyCount, const size_t targetUniqueTokens)
{
        std::mutex frequencyCountMutex;
        std::unordered_set<std::string> usedTokens;
        std::unordered_map<std::string, size_t> checkedTokens;
        std::priority_queue<std::pair<size_t, std::string>> checkedTokensPriorityQueue;
        uint8_t numThreads = std::thread::hardware_concurrency();

        std::vector<std::unordered_map<std::string, size_t>> threadSpecificFrequencyCounts(numThreads);
        size_t i = 0;
        for (const auto& token : frequencyCount)
        {
                threadSpecificFrequencyCounts[i % numThreads][token.first] = token.second;
                i++;
        }

        while (frequencyCount.size() < targetUniqueTokens)
        {
                std::vector<std::thread> threads;
                for (uint8_t i = 0; i < numThreads; i++)
                {
                        threads.push_back(std::thread(calculateFrequencyThread,
                                                      std::ref(frequencyCount),
                                                      std::ref(threadSpecificFrequencyCounts[i]),
                                                      std::ref(wordsByLength),
                                                      std::ref(checkedTokens),
                                                      std::ref(checkedTokensPriorityQueue),
                                                      std::ref(frequencyCountMutex)));
                }

                for (auto& thread : threads)
                {
                        thread.join();
                }

                for (uint8_t i = 0; i < numThreads; i++)
                {
                        for (const auto& token : threadSpecificFrequencyCounts[i])
                        {
                                frequencyCount[token.first] = token.second;
                        }
                }

                std::string mostFrequentUnusedToken;
                size_t newFrequency = 0;
                while (true)
                {
                        if (checkedTokensPriorityQueue.empty())
                        {
                                break;
                        }
                        std::pair<size_t, std::string> mostFrequentToken = checkedTokensPriorityQueue.top();
                        checkedTokensPriorityQueue.pop();
                        if (usedTokens.find(mostFrequentToken.second) == usedTokens.end())
                        {
                                mostFrequentUnusedToken = mostFrequentToken.second;
                                newFrequency = mostFrequentToken.first;
                                break;
                        }
                }
                if (newFrequency == 0)
                {
                        std::cout << "No more tokens to add\n";
                        break;
                }

                usedTokens.insert(mostFrequentUnusedToken);
                frequencyCount[mostFrequentUnusedToken] = newFrequency;
                LOG("Added token: " + mostFrequentUnusedToken + " with frequency: " + std::to_string(newFrequency));
                LOG("Checked tokens: " + std::to_string(checkedTokens.size()));
                LOG("Unique tokens: " + std::to_string(frequencyCount.size()));
        }

        if (frequencyCount.size() == targetUniqueTokens)
                std::cout << "Reached target unique tokens\n";
        else
                std::cout << "Did not reach target unique tokens\n";
}

void tokenizeDataset(const std::string& inputPath, const std::string& outputPath)
{
        std::ifstream inputFile(inputPath);
        std::string line;
        std::unordered_map<size_t, std::unordered_set<std::string>> wordsByLength;
        std::unordered_map<std::string, size_t> frequencyCount = {
                {"a", 0},
                {"b", 0},
                {"c", 0},
                {"d", 0},
                {"e", 0},
                {"f", 0},
                {"g", 0},
                {"h", 0},
                {"i", 0},
                {"j", 0},
                {"k", 0},
                {"l", 0},
                {"m", 0},
                {"n", 0},
                {"o", 0},
                {"p", 0},
                {"q", 0},
                {"r", 0},
                {"s", 0},
                {"t", 0},
                {"u", 0},
                {"v", 0},
                {"w", 0},
                {"x", 0},
                {"y", 0},
                {"z", 0},
                {" ", 0},
                {".", 0},
                {"!", 0},
                {"?", 0},
                {",", 0},
                {";", 0},
                {":", 0},
                {"+", 0},
                {"=", 0}};

        std::unordered_map<char, std::string> charToString = {
                {'a', "a"},
                {'b', "b"},
                {'c', "c"},
                {'d', "d"},
                {'e', "e"},
                {'f', "f"},
                {'g', "g"},
                {'h', "h"},
                {'i', "i"},
                {'j', "j"},
                {'k', "k"},
                {'l', "l"},
                {'m', "m"},
                {'n', "n"},
                {'o', "o"},
                {'p', "p"},
                {'q', "q"},
                {'r', "r"},
                {'s', "s"},
                {'t', "t"},
                {'u', "u"},
                {'v', "v"},
                {'w', "w"},
                {'x', "x"},
                {'y', "y"},
                {'z', "z"},
                {' ', " "},
                {'.', "."},
                {'!', "!"},
                {'?', "?"},
                {',', ","},
                {';', ";"},
                {':', ":"},
                {'+', "+"},
                {'=', "="}};

        while (std::getline(inputFile, line))
        {
                std::istringstream ss(line);
                std::string word;
                bool firstWord = true;
                while (std::getline(ss, word, ' '))
                {
                        if (!firstWord)
                        {
                                word = " " + word;
                        }
                        size_t wordLength = word.size();
                        if (wordLength > MAX_WORD_LENGTH)
                        {
                                continue;
                        }
                        wordsByLength[wordLength].insert(word);
                        for (char c : word)
                        {
                                frequencyCount[charToString[c]]++;
                        }
                        firstWord = false;
                }
        }
        inputFile.close();

        addTokens(wordsByLength, frequencyCount, TARGET_UNIQUE_TOKENS);

        std::ofstream outputFile(outputPath);
        for (const auto& token : frequencyCount)
        {
                outputFile << token.first << "\n";
        }
        outputFile.close();

        std::cout << "Dataset tokenized\n";
}

TokenDictionary loadTokenDictionary(const std::string& inputPath)
{
        TokenDictionary tokenDictionary;
        std::ifstream inputFile(inputPath);
        std::string line;
        uint16_t tokenIndex = 0;
        while (std::getline(inputFile, line))
        {
                tokenDictionary.tokenToIndex[line] = tokenIndex;
                tokenDictionary.indexToToken[tokenIndex] = line;
                tokenDictionary.tokensByLength.push(std::make_pair(line.size(), line));
                tokenIndex++;
        }
        inputFile.close();
        return tokenDictionary;
}

std::vector<uint16_t> tokenizeString(const std::string& input, TokenDictionary& tokenDictionary)
{
        std::vector<uint16_t> tokens;
        std::string inputCopy = input;

        while (!inputCopy.empty())
        {
                std::priority_queue<std::pair<size_t, std::string>> tokensByLengthCopy = tokenDictionary.tokensByLength;
                size_t length = inputCopy.size();
                while (!tokensByLengthCopy.empty())
                {
                        std::pair<size_t, std::string> token = tokensByLengthCopy.top();
                        tokensByLengthCopy.pop();
                        if (length < token.first)
                        {
                                continue;
                        }
                        if (inputCopy.find(token.second) == 0)
                        {
                                tokens.push_back(tokenDictionary.tokenToIndex[token.second]);
                                inputCopy = inputCopy.substr(token.second.size());
                                if (inputCopy.empty())
                                {
                                        break;
                                }
                                else
                                {
                                        tokensByLengthCopy = tokenDictionary.tokensByLength;
                                }
                        }
                }
        }

        return tokens;
}

std::string detokenizeString(const std::vector<uint16_t>& tokens, TokenDictionary& tokenDictionary)
{
        std::string detokenized;

        for (uint16_t token : tokens)
        {
                detokenized += tokenDictionary.indexToToken[token] + " ";
        }

        return detokenized;
}
