#include <iostream>

#include <cicero/cicero.h>
#include "matrix.hpp"
#include "vector.hpp"

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


        const uint64_t N = 2;
        Matrix<float, N, N> matrix(2.0f);
        Matrix<float, N, N> diagonal;
        diagonal.diagonal(5.0f);
        matrix*=diagonal;
        matrix.print();
        std::cout << matrix.trace() << "\n";

        Vector<float, 3> vector(1.0f);
        vector.print();
        vector.normalize().print();

        std::cout << vector.magnitude() << "\n";
        std::cout << vector.dot(vector) << "\n";
        vector.cross(vector).print();

        Vector<float, 3> cross1(1.0f, 0.0f, 0.0f);
        Vector<float, 3> cross2(0.0f, 1.0f, 0.0f);

        cross2.cross(cross1).print();

        return 0;
}
