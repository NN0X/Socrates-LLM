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

// example amazon review data
// "2","header","text"\n
// "1","header","text"\n
// "5","header","text"\n

const std::vector<char> PUNCTUATION = {'.', '!', '?', ',', ';', ':', '+', '='};

void prepareAmazonReviewData(const std::string& inputPath, const std::string& outputPath)
{
        std::ifstream inputFile(inputPath);
        std::string line;
        std::ofstream outputFile(outputPath);
        while (std::getline(inputFile, line))
        {
                std::vector<std::string> parts;
                std::string part;
                std::istringstream ss(line);
                while (std::getline(ss, part, ','))
                {
                        parts.push_back(part);
                }

                std::string temp = parts[2];
                for (size_t i = 0; i < temp.size(); i++)
                {
                        char c = temp[i];
                        if (c == '"' || c == '\n' || c == '\r' || c == '\t')
                        {
                                c = ' ';
                        }
                        if (c >= 'A' && c <= 'Z')
                        {
                                c = c - 'A' + 'a';
                        }
                        temp[i] = c;
                        if (std::find(PUNCTUATION.begin(), PUNCTUATION.end(), c) != PUNCTUATION.end())
                        {
                                if (i + 1 < temp.size() && temp[i + 1] != ' ' && std::find(PUNCTUATION.begin(), PUNCTUATION.end(), temp[i + 1]) == PUNCTUATION.end())
                                {
                                        temp.insert(i + 1, " ");
                                        i++;
                                }
                        }
                }
                outputFile << temp << std::endl;
        }

        inputFile.close();
        outputFile.close();
}

#define MAX_WORD_LENGTH 45
#define TARGET_UNIQUE_TOKENS 1000

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
                                for (const auto& word : wordsLengthBracket.second)
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
                std::cout << "Added token: " << mostFrequentUnusedToken << " with frequency: " << newFrequency << "\n";
                std::cout << "Checked tokens: " << checkedTokens.size() << "\n";
                std::cout << "Unique tokens: " << frequencyCount.size() << "\n";
        }

        if (frequencyCount.size() < targetUniqueTokens)
                std::cout << "Reached target unique tokens\n";
}

void tokenize(const std::string& inputPath, const std::string& outputPath)
{
        std::ifstream inputFile(inputPath);
        std::string line;
        std::unordered_map<size_t, std::unordered_set<std::string>> wordsByLength;
        std::unordered_map<std::string, size_t> frequencyCount;
        while (std::getline(inputFile, line))
        {
                std::istringstream ss(line);
                std::string word;
                while (std::getline(ss, word, ' '))
                {
                        if (word.size() > MAX_WORD_LENGTH)
                        {
                                continue;
                        }
                        wordsByLength[word.size()].insert(word);
                        for (char c : word)
                        {
                                if (frequencyCount.find(std::string(1, c)) == frequencyCount.end())
                                {
                                        frequencyCount[std::string(1, c)] = 1;
                                }
                                else
                                {
                                        frequencyCount[std::string(1, c)]++;
                                }
                        }
                }
        }

        addTokens(wordsByLength, frequencyCount, TARGET_UNIQUE_TOKENS);
        std::cout << "Unique tokens: " << frequencyCount.size() << "\n";
}
