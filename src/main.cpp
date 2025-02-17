#include <iostream>

#include <cicero/cicero.h>
#include "matrix.hpp"

// TODO: use HIP for GPU acceleration
// TODO: use "Attention is All You Need" in GPT architecture
// TODO: use Adam optimizer to train the model (gradient descent)

int main()
{
        //TokenDictionary tokenDictionary = loadTokenDictionary("resources/2^16.tok");
        //std::string input = "This is a test string.";
        //std::vector<uint16_t> tokens = tokenizeString(input, tokenDictionary);

        //for (uint16_t token : tokens)
        //{
        //        std::cout << token << " ";
        //}
        //std::cout << "\n";

        //std::string output = detokenizeString(tokens, tokenDictionary);
        //std::cout << output << "\n";

        Matrix<int, 2, 2> matrix1;

        matrix1(0, 0) = 1;
        matrix1(0, 1) = 2;
        matrix1(1, 0) = 3;

        Matrix<int, 2, 2> matrix2;
        matrix2(0, 0) = 4;
        matrix2(0, 1) = 5;
        matrix2(1, 0) = 6;

        Matrix<int, 2, 2> matrix3 = matrix1 + matrix2;

        matrix1.print();
        std::cout << "\n";
        matrix2.print();
        std::cout << "\n";
        matrix3.print();

        matrix3 *= 2;
        std::cout << "\n";
        matrix3.print();

        matrix3 *= matrix1;
        std::cout << "\n";
        matrix3.print();

        matrix3 = matrix1 * matrix2;
        std::cout << "\n";
        matrix3.print();

        return 0;
}
