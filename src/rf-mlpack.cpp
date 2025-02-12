#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

#include <mlpack.hpp>

#define MLPACK_PRINT_INFO
#define MLPACK_PRINT_WARN

#define MAX_COLUMNS 300000
#define MAX_DEPTH 10

void test()
{
        arma::mat dataset;
        arma::Row<size_t> labels;
        if (!mlpack::data::Load("resources/covertype-small.data.csv", dataset))
          throw std::runtime_error("Could not read covertype-small.data.csv!");
        if (!mlpack::data::Load("resources/covertype-small.labels.csv", labels))
          throw std::runtime_error("Could not read covertype-small.labels.csv!");

        // print number of rows and columns
        std::cout << "Number of rows: " << dataset.n_rows << std::endl;
        std::cout << "Number of columns: " << dataset.n_cols << std::endl;

        labels -= 1;

        // Now split the dataset into a training set and test set, using 30% of the
        // dataset for the test set.
        arma::mat trainDataset, testDataset;
        arma::Row<size_t> trainLabels, testLabels;
        mlpack::data::Split(dataset, labels, trainDataset, testDataset, trainLabels,
            testLabels, 0.3);

        // Create the RandomForest object and train it on the training data.
        mlpack::tree::RandomForest<> r(trainDataset,
                         trainLabels,
                         7 /* number of classes */,
                         10 /* number of trees */,
                         3 /* minimum leaf size */);

        // Compute and print the training error.
        arma::Row<size_t> trainPredictions;
        r.Classify(trainDataset, trainPredictions);
        const double trainError =
            arma::accu(trainPredictions != trainLabels) * 100.0 / trainLabels.n_elem;
        std::cout << "Training error: " << trainError << "%." << std::endl;

        // Now compute predictions on the test points.
        arma::Row<size_t> testPredictions;
        r.Classify(testDataset, testPredictions);
        const double testError =
            arma::accu(testPredictions != testLabels) * 100.0 / testLabels.n_elem;
        std::cout << "Test error: " << testError << "%." << std::endl;
}

//i source data example:
// 0,This is a example sentence
// 1,This is another example sentence
// 2,This is a third example sentence
// 3,This is a fourth example sentence
// 4,This is a fifth example sentence
//
// mapping example:
// This,is,a,example,sentence,another,third,fourth
//
// output example:
//
// dataset:     labels:
// 1,2,3,4,5,0  0
// 1,2,6,4,5,0  0
// 1,2,3,7,4,5  0
// 1,2,3,8,4,5  0
// 1,2,3,0,4,5  0
// 1,-1,3,4,5,0 2
// 1,2,-1,4,5,0 3
// 1,2,-1,4,5,0 6
// 1,2,3,-1,4,5 7
// 1,2,3,-1,4,5 8
// and so on

// UINT64_MAX is reserved for padding
// UINT64_MAX - 1 is reserved for unknown words
// UINT64_MAX - 2 is reserved for the label placement

#define EMPTY 0
#define UNKNOWN_LABEL 1
#define UNKNOWN_DATA UINT64_MAX - 1
#define LABEL UINT64_MAX
#define CSV_DELIMITER ','

std::vector<size_t> mapSentence(const std::string &sentence, const std::unordered_map<std::string, size_t> &mapping)
{
        std::vector<size_t> result;
        std::vector<std::string> words;
        std::stringstream ss(sentence);
        std::string token;
        while (std::getline(ss, token, ' '))
        {
                words.push_back(token);
        }
        for (const std::string &word : words)
        {
                if (word == "___")
                {
                        result.push_back(LABEL);
                }
                else if (mapping.count(word) > 0)
                {
                        result.push_back(mapping.at(word));
                }
                else
                {
                        result.push_back(UNKNOWN_DATA);
                }
        }
        return result;
}

struct DatasetTranslation
{
        std::unordered_map<std::string, size_t> mapping;
        std::unordered_map<size_t, std::string> reverseMapping;
        std::vector<std::vector<size_t>> dataset;
        std::vector<size_t> labels;
        size_t uniqueLabels;
        size_t maxSentenceLength;
};

DatasetTranslation prepareData(const std::string &sourcePath, const std::string &mappingPath,const std::string &datasetPath, const std::string &labelsPath)
{
        DatasetTranslation result;
        result.mapping["<unknown>"] = UNKNOWN_LABEL;
        result.mapping[""] = EMPTY;
        result.reverseMapping[EMPTY] = "";
        result.reverseMapping[UNKNOWN_LABEL] = "<unknown>";
        result.uniqueLabels = 2;

        std::vector<std::vector<size_t>> baseSentences;

        std::ifstream mappingFile(mappingPath);
        if (mappingFile.is_open())
        {
                std::string line;
                while (std::getline(mappingFile, line))
                {
                        std::stringstream ss(line);
                        std::string token;
                        size_t index = 2;
                        while (std::getline(ss, token, CSV_DELIMITER))
                        {
                                result.mapping[token] = index;
                                result.reverseMapping[index] = token;
                                index++;
                                result.uniqueLabels++;
                        }
                }
        }
        mappingFile.close();

        std::ifstream sourceFile(sourcePath);
        if (sourceFile.is_open())
        {
                std::string line;
                while (std::getline(sourceFile, line))
                {
                        size_t pos = line.find(CSV_DELIMITER);
                        if (pos != std::string::npos)
                        {
                                line = line.substr(pos + 1);
                                baseSentences.push_back(mapSentence(line, result.mapping));
                        }
                }
        }
        sourceFile.close();

        size_t longest = 0;
        for (const std::vector<size_t> &sentence : baseSentences)
        {
                if (sentence.size() > longest)
                {
                        longest = sentence.size();
                }
        }
        for (std::vector<size_t> &sentence : baseSentences)
        {
                while (sentence.size() < longest)
                {
                        sentence.push_back(EMPTY);
                }
        }
        result.maxSentenceLength = longest;

        // create gap sentences

        size_t count = 0;
        for (const std::vector<size_t> &sentence : baseSentences)
        {
                size_t sentenceLength = std::find(sentence.begin(), sentence.end(), EMPTY) - sentence.begin();
                for (size_t i = 0; i < sentenceLength; i++)
                {
                        if (sentence[i] == UNKNOWN_DATA)
                        {
                                continue;
                        }
                        std::vector<size_t> gapSentence = sentence;
                        gapSentence[i] = LABEL;
                        result.dataset.push_back(gapSentence);
                        result.labels.push_back(sentence[i]);
                        count++;
                        if (count == MAX_COLUMNS)
                        {
                                break;
                        }
                }
                if (count == MAX_COLUMNS)
                {
                        break;
                }
        }

        std::ofstream datasetFile(datasetPath);
        if (datasetFile.is_open())
        {
                for (const std::vector<size_t> &sentence : result.dataset)
                {
                        for (size_t i = 0; i < sentence.size(); i++)
                        {
                                datasetFile << sentence[i];
                                if (i < sentence.size() - 1)
                                {
                                        datasetFile << CSV_DELIMITER;
                                }
                        }
                        datasetFile << std::endl;
                }
        }
        datasetFile.close();

        std::ofstream labelsFile(labelsPath);
        if (labelsFile.is_open())
        {
                for (size_t label : result.labels)
                {
                        labelsFile << label << std::endl;
                }
        }
        labelsFile.close();

        return result;
}

int main()
{
        //test();


        DatasetTranslation data = prepareData("resources/cv-unique-no-end-punct-sentences.csv",
                                              "resources/en-dictionary.csv", "resources/dataset.csv", "resources/labels.csv");

        std::ifstream modelFile("bin/model.bin");
        if (modelFile.good())
        {
                mlpack::tree::RandomForest<> r;
                mlpack::data::Load("bin/model.bin", "model", r);
                std::cout << "Model loaded successfully!" << std::endl;
                std::string sentence;
                while (sentence != "/bye")
                {
                        std::cout << "Enter a sentence: ";
                        std::getline(std::cin, sentence);
                        std::vector<size_t> sentenceVector = mapSentence(sentence, data.mapping);
                        while (sentenceVector.size() < data.maxSentenceLength)
                        {
                                sentenceVector.push_back(EMPTY);
                        }
                        arma::vec point = arma::conv_to<arma::vec>::from(arma::Row<size_t>(sentenceVector));
                        size_t prediction = r.Classify(point);
                        std::cout << "Prediction: " << data.reverseMapping[prediction] << std::endl;
                }
                return 0;
        }

        arma::mat dataset;
        arma::Row<size_t> labels;
        if (!mlpack::data::Load("resources/dataset.csv", dataset))
                throw std::runtime_error("Could not read dataset.csv!");
        if (!mlpack::data::Load("resources/labels.csv", labels))
                throw std::runtime_error("Could not read labels.csv!");

        std::cout << "Number of rows: " << dataset.n_rows << std::endl;
        std::cout << "Number of columns: " << dataset.n_cols << std::endl;

        arma::mat trainDataset, testDataset;
        arma::Row<size_t> trainLabels, testLabels;
        mlpack::data::Split(dataset, labels, trainDataset, testDataset, trainLabels,
            testLabels, 0.3);

        mlpack::tree::RandomForest<> r(trainDataset,
                                       trainLabels,
                                       data.uniqueLabels /* number of classes */,
                                       20 /* number of trees */,
                                       3 /* minimum leaf size */, 
                                       1e-7 /* minimum gain split */,
                                       MAX_DEPTH /* maximum depth */
                                       );

        arma::Row<size_t> trainPredictions;
        r.Classify(trainDataset, trainPredictions);
        const double trainError =
            arma::accu(trainPredictions != trainLabels) * 100.0 / trainLabels.n_elem;
        std::cout << "Training error: " << trainError << "%." << std::endl;

        // Now compute predictions on the test points.
        arma::Row<size_t> testPredictions;
        r.Classify(testDataset, testPredictions);
        const double testError =
            arma::accu(testPredictions != testLabels) * 100.0 / testLabels.n_elem;
        std::cout << "Test error: " << testError << "%." << std::endl;

        // save r to binary file
        mlpack::data::Save("bin/model.bin", "model", r);

        return 0;
}
