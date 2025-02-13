#include <iostream>

#include "datasets.h"
#include "tokenizer.h"

// TODO: use HIP for GPU acceleration
// TODO: tokenize training and test datasets using subword tokenizer into training.tok and test.tok
// TODO: use "Attention is All You Need" in GPT architecture
// TODO: use Adam optimizer to train the model (gradient descent)

int main()
{
        prepareAmazonReviewData("resources/test.csv", "resources/test_prepared.txt");
        prepareAmazonReviewData("resources/train.csv", "resources/train_prepared.txt");

        tokenizeDataset("resources/test_prepared.txt", "resources/2^15.tok");
        TokenDictionary tokenDictionary = loadTokenDictionary("resources/2^15.tok");

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
