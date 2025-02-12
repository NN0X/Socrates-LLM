#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <climits>

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

void addPhrases(const std::map<size_t, std::set<std::string>> &wordsByLength, std::map<std::string, size_t>& frequencyCount, const size_t targetUniqueTokens)
{
        std::set<std::string> usedPhrases;
        std::map<std::string, size_t> checkedPhrases;
        std::string mostFrequentUnusedPhrase;
        while (frequencyCount.size() < targetUniqueTokens)
        {
                // find most frequent phrase
                std::string newPhrase;
                size_t newFrequency = 0;

                // multithread this for loop
                for (auto phrase1 : frequencyCount)
                {
                        for (auto phrase2 : frequencyCount)
                        {
                                std::string phrase = phrase1.first + phrase2.first;
                                if (phrase.size() > MAX_WORD_LENGTH || usedPhrases.find(phrase) != usedPhrases.end() || checkedPhrases.find(phrase) != checkedPhrases.end())
                                {
                                        continue;
                                }
                                size_t frequency = 0;
                                for (auto sizeBracket : wordsByLength)
                                {
                                        if (sizeBracket.first < phrase.size())
                                        {
                                                continue;
                                        }
                                        for (auto word : sizeBracket.second)
                                        {
                                                if (word.find(phrase) != std::string::npos)
                                                {
                                                        frequency++;
                                                }
                                        }
                                }
                                if (frequency > 0)
                                        std::cout << "Potential new phrase: " << phrase << " with frequency: " << frequency << "\n";
                                checkedPhrases[phrase] = frequency;

                                // critical section
                                if (frequency > newFrequency)
                                {
                                        newPhrase = phrase;
                                        newFrequency = frequency;
                                }
                                // end critical section
                        }
                }

                std::string newMostFrequentUnusedPhrase;
                for (auto phrase : checkedPhrases)
                {
                        if (usedPhrases.find(phrase.first) == usedPhrases.end())
                        {
                                if (newMostFrequentUnusedPhrase.empty() || phrase.second > checkedPhrases[newMostFrequentUnusedPhrase])
                                {
                                        newMostFrequentUnusedPhrase = phrase.first;
                                }
                        }
                }
                if (!newMostFrequentUnusedPhrase.empty() && checkedPhrases[newMostFrequentUnusedPhrase] > newFrequency)
                {
                        newPhrase = newMostFrequentUnusedPhrase;
                        newFrequency = checkedPhrases[newMostFrequentUnusedPhrase];
                }
                if (!mostFrequentUnusedPhrase.empty() && checkedPhrases[mostFrequentUnusedPhrase] > newFrequency)
                {
                        std::string temp = newPhrase;
                        newPhrase = mostFrequentUnusedPhrase;
                        newFrequency = checkedPhrases[mostFrequentUnusedPhrase];
                        mostFrequentUnusedPhrase = temp;
                }
                if (newFrequency == 0)
                {
                        std::cout << "No more phrases to add\n";
                        break;
                }

                usedPhrases.insert(newPhrase);
                frequencyCount[newPhrase] = newFrequency;
                std::cout << "Added phrase: " << newPhrase << " with frequency: " << newFrequency << "\n";
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
