#include <iostream>

#include "tokenizer.h"


// TODO: use HIP for GPU acceleration
// TODO: cleanup amazon reviews dataset (remove scores, split up reviews into sentences)
// TODO: cleanup amazon reviews test dataset (remove scores, split up reviews into sentences, create gaps for model to fill in)
// TODO: create a subword tokenizer
// TODO: tokenize training and test datasets using subword tokenizer into training.tok and test.tok
// TODO: use "Attention is All You Need" in GPT architecture
// TODO: use Adam optimizer to train the model (gradient descent)

int main()
{
        //prepareAmazonReviewData("resources/test.csv", "resources/test_prepared.txt");
        tokenize("resources/test_prepared.txt", "resources/test.tok");
}
