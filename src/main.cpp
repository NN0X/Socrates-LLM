#include <iostream>

#include "tokenizer.h"


// TODO: use HIP for GPU acceleration
// TODO: create a subword tokenizer
// TODO: tokenize training and test datasets using subword tokenizer into training.tok and test.tok
// TODO: use "Attention is All You Need" in GPT architecture
// TODO: use Adam optimizer to train the model (gradient descent)

int main()
{
        tokenizeDataset("resources/test_prepared.txt", "resources/test.tok");
}
