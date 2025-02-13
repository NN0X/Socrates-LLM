#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <cstdint>

// example amazon review data
// "2","header","text"\n
// "1","header","text"\n
// "5","header","text"\n

const std::vector<char> PUNCTUATION = {'.', '!', '?', ',', ';', ':', '+', '='};

void mergeDatasets(const std::vector<std::string>& datasetPaths, const std::string& outputPath)
{
        std::ofstream outputFile(outputPath);
        for (const std::string& datasetPath : datasetPaths)
        {
                std::ifstream inputFile(datasetPath);
                std::string line;
                while (std::getline(inputFile, line))
                {
                        outputFile << line << "\n";
                }
                inputFile.close();
        }
        outputFile.close();
}

void prepareAmazonReviewDataThread(const std::vector<std::string>& dataPart, const std::string& outputPath)
{
        std::ofstream outputFile(outputPath);
        for (const std::string& line : dataPart)
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
                                temp.erase(i, 1);
                                i--;
                                continue;
                        }
                        //if (c >= 'A' && c <= 'Z')
                        //{
                        //        c = c - 'A' + 'a';
                        //}
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
                outputFile << temp << "\n";
        }
        outputFile.close();
}

void prepareAmazonReviewData(const std::string& inputPath, const std::string& outputPath)
{
        uint8_t numThreads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads(numThreads);

        size_t numLines = 0;
        std::ifstream inputFile(inputPath);
        std::string line;
        std::vector<std::string> lines;
        while (std::getline(inputFile, line))
        {
                lines.push_back(line);
                numLines++;
        }
        inputFile.close();

        size_t linesPerThread = numLines / numThreads;
        size_t currentLine = 0;

        std::vector<std::vector<std::string>> dataParts(numThreads);
        for (uint8_t i = 0; i < numThreads; i++)
        {
                for (size_t j = 0; j < linesPerThread; j++)
                {
                        dataParts[i].push_back(lines[currentLine]);
                        currentLine++;
                }
        }

        std::vector<std::string> outputPaths(numThreads);
        for (uint8_t i = 0; i < numThreads; i++)
        {
                std::string outputPathPart = outputPath + "." + std::to_string(i) + ".part";
                outputPaths[i] = outputPathPart;
                threads[i] = std::thread(prepareAmazonReviewDataThread, std::ref(dataParts[i]), outputPathPart);
        }

        for (uint8_t i = 0; i < numThreads; i++)
        {
                threads[i].join();
        }

        mergeDatasets(outputPaths, outputPath);

        for (uint8_t i = 0; i < numThreads; i++)
        {
                std::remove(outputPaths[i].c_str());
        }
}

