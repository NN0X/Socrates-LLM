#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <climits>
#include <thread>
#include <mutex>
#include <cstdint>
#include <queue>

// TODO: compare unoredered_map with map for performance (checkedPhrases)
// TODO: compare unordered_set with set for performance (usedPhrases)

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
#define TARGET_UNIQUE_TOKENS 50000

void calculateFrequencyThread(std::map<std::string, size_t>& frequencyCount,
                              std::map<std::string, size_t>& threadSpecificFrequencyCount,
                              const std::map<size_t, std::set<std::string>>& wordsByLength,
                              std::unordered_map<std::string, size_t>& checkedPhrases,
                              std::priority_queue<std::pair<size_t, std::string>>& checkedPhrasesPriorityQueue,
                              std::mutex& frequencyCountMutex)
{
        for (const auto& phrase1 : threadSpecificFrequencyCount)
        {
                for (const auto& phrase2 : frequencyCount)
                {
                        std::string phrase = phrase1.first + phrase2.first;
                        if (phrase.size() > MAX_WORD_LENGTH || checkedPhrases.find(phrase) != checkedPhrases.end())
                        {
                                continue;
                        }
                        size_t frequency = 0;
                        for (const auto& sizeBracket : wordsByLength)
                        {
                                if (sizeBracket.first < phrase.size())
                                {
                                        continue;
                                }
                                for (const auto& word : sizeBracket.second)
                                {
                                        if (word.find(phrase) != std::string::npos)
                                        {
                                                frequency++;
                                        }
                                }
                        }

                        {
                                std::lock_guard<std::mutex> lock(frequencyCountMutex);
                                checkedPhrases[phrase] = frequency;
                                checkedPhrasesPriorityQueue.push(std::make_pair(frequency, phrase));
                        }
                }
        }
}

void addPhrases(const std::map<size_t, std::set<std::string>> &wordsByLength, std::map<std::string, size_t>& frequencyCount, const size_t targetUniqueTokens)
{
        std::mutex frequencyCountMutex;
        std::unordered_set<std::string> usedPhrases;
        std::unordered_map<std::string, size_t> checkedPhrases;
        std::priority_queue<std::pair<size_t, std::string>> checkedPhrasesPriorityQueue;
        uint8_t numThreads = std::thread::hardware_concurrency();

        std::vector<std::map<std::string, size_t>> threadSpecificFrequencyCounts(numThreads);
        size_t i = 0;
        for (const auto& phrase : frequencyCount)
        {
                threadSpecificFrequencyCounts[i % numThreads][phrase.first] = phrase.second;
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
                                                      std::ref(checkedPhrases),
                                                      std::ref(checkedPhrasesPriorityQueue),
                                                      std::ref(frequencyCountMutex)));
                }

                for (auto& thread : threads)
                {
                        thread.join();
                }

                for (uint8_t i = 0; i < numThreads; i++)
                {
                        for (const auto& phrase : threadSpecificFrequencyCounts[i])
                        {
                                frequencyCount[phrase.first] = phrase.second;
                        }
                }

                std::string mostFrequentUnusedPhrase;
                size_t newFrequency = 0;
                while (true)
                {
                        if (checkedPhrasesPriorityQueue.empty())
                        {
                                break;
                        }
                        std::pair<size_t, std::string> mostFrequentPhrase = checkedPhrasesPriorityQueue.top();
                        checkedPhrasesPriorityQueue.pop();
                        if (usedPhrases.find(mostFrequentPhrase.second) == usedPhrases.end())
                        {
                                mostFrequentUnusedPhrase = mostFrequentPhrase.second;
                                newFrequency = mostFrequentPhrase.first;
                                break;
                        }
                }
                if (newFrequency == 0)
                {
                        std::cout << "No more phrases to add\n";
                        break;
                }

                usedPhrases.insert(mostFrequentUnusedPhrase);
                frequencyCount[mostFrequentUnusedPhrase] = newFrequency;
                std::cout << "Added phrase: " << mostFrequentUnusedPhrase << " with frequency: " << newFrequency << "\n";
                std::cout << "Checked phrases: " << checkedPhrases.size() << "\n";
                std::cout << "Unique tokens: " << frequencyCount.size() << "\n";
        }

        if (frequencyCount.size() < targetUniqueTokens)
                std::cout << "Reached target unique tokens\n";
}

void tokenize(const std::string& inputPath, const std::string& outputPath)
{
        std::ifstream inputFile(inputPath);
        std::string line;
        std::map<size_t, std::set<std::string>> wordsByLength;
        std::map<std::string, size_t> frequencyCount;
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

        addPhrases(wordsByLength, frequencyCount, TARGET_UNIQUE_TOKENS);
        std::cout << "Unique tokens: " << frequencyCount.size() << "\n";
}
