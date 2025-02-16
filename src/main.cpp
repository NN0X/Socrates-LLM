#include <iostream>

#include <cicero/cicero.h>

// TODO: use HIP for GPU acceleration
// TODO: tokenize training and test datasets using subword tokenizer into training.tok and test.tok
// TODO: use "Attention is All You Need" in GPT architecture
// TODO: use Adam optimizer to train the model (gradient descent)

int main()
{
        TokenDictionary tokenDictionary = loadTokenDictionary("dictionaries/2^16.tok");
        std::string input = "This is a test string.";
        std::vector<uint16_t> tokens = tokenizeString(input, tokenDictionary);

        for (uint16_t token : tokens)
        {
                std::cout << token << " ";
        }
        std::cout << "\n";

        std::string output = detokenizeString(tokens, tokenDictionary);
        std::cout << output << "\n";

        return 0;
}
