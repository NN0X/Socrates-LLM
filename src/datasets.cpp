#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>


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
                outputFile << temp << "\n";
        }

        inputFile.close();
        outputFile.close();
}

