#include <iostream>

#include "tokenizer.h"

// TODO: use HIP for GPU acceleration
// TODO: tokenize training and test datasets using subword tokenizer into training.tok and test.tok
// TODO: use "Attention is All You Need" in GPT architecture
// TODO: use Adam optimizer to train the model (gradient descent)

int main()
{
        //tokenizeDataset("resources/test_prepared.txt", "resources/test.tok");
        TokenDictionary tokenDictionary = loadTokenDictionary("resources/test.tok");

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
