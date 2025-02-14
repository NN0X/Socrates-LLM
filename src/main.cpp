#include <iostream>

#include "datasets.h"
#include "tokenizer.h"

// TODO: use HIP for GPU acceleration
// TODO: tokenize training and test datasets using subword tokenizer into training.tok and test.tok
// TODO: use "Attention is All You Need" in GPT architecture
// TODO: use Adam optimizer to train the model (gradient descent)

int main()
{
        //prepareAmazonReviewData("resources/test.csv", "resources/test_prepared.txt");
        //prepareAmazonReviewData("resources/train.csv", "resources/train_prepared.txt");

        //std::vector<std::string> datasets = {"resources/train_prepared.txt", "resources/test_prepared.txt"};
        //mergeDatasets(datasets, "resources/all.txt");

        //tokenizeDataset("resources/all.txt", "resources/2^16.tok");
        TokenDictionary tokenDictionary = loadTokenDictionary("dictionaries/2^16.tok");

        // test tokenization
        std::vector<uint16_t> tokens = tokenizeString("hello world", tokenDictionary);
        for (uint16_t token : tokens)
        {
                std::cout << token << " ";
        }
        std::cout << "\n";

        std::string detokenized = detokenizeString(tokens, tokenDictionary);
        std::cout << detokenized << "\n";

        return 0;
}
