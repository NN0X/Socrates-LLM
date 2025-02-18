// TODO: use Strassen Algorithm for big matrix multiplication
// TODO: check the matrix sizes at which Strassen Algorithm is faster than normal multiplication

// INFO: This is the CPU matrix class implementation.

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <cstdint>
#include <thread>
#include <vector>
#include <stdexcept>
#include <array>
#include <memory>
#include <type_traits>

#include "heaparray.hpp"

#define THREADING_THRESHOLD 500 // INFO: threading is used only when R > THREADING_THRESHOLD
#define BIG_MATRIX_SIZE 1'000'000'000 // INFO: matrix size at which big matrix algorithms are used

template <typename T, uint64_t R, uint64_t C, typename Enable = void>
class Matrix;

template <typename T, uint64_t R, uint64_t C>
using ThreadFunction = void(*)(Matrix<T, R, C> &, const Matrix<T, R, C> &, const Matrix<T, R, C> &, uint64_t, uint64_t);

// TODO: move dispatchThreads to a specific functions and rename e.g. dispatchThreadsMultiply

template <typename T, uint64_t R, uint64_t C>
void dispatchThreads(Matrix<T, R, C> &result,
                     const Matrix<T, R, C> &matrix1,
                     const Matrix<T, R, C> &matrix2,
                     ThreadFunction<T, R, C> threadFunction)
{
        if (R < std::thread::hardware_concurrency() * 100)
        {
                threadFunction(result, matrix1, matrix2, 0, R);
                return;
        }
        std::vector<std::thread> threads;
        uint8_t numThreads = std::thread::hardware_concurrency();
        threads.reserve(numThreads);
        uint64_t chunkSize = R / numThreads;
        for (uint8_t i = 0; i < numThreads; i++)
        {
                if (i == numThreads - 1)
                        threads.emplace_back(threadFunction, std::ref(result), std::ref(matrix1), std::ref(matrix2), i*chunkSize, R);
                else
                        threads.emplace_back(threadFunction, std::ref(result), std::ref(matrix1), std::ref(matrix2), i*chunkSize, (i + 1)*chunkSize);
        }
        for (uint8_t i = 0; i < numThreads; i++)
                threads[i].join();
}

template <typename T, uint64_t R, uint64_t C, uint64_t C2>
void dispatchThreads(Matrix<T, R, C> &result,
                     const Matrix<T, R, C> &matrix1,
                     const Matrix<T, C, C2> &matrix2,
                     ThreadFunction<T, R, C> threadFunction)
{
        if (R < THREADING_THRESHOLD)
        {
                threadFunction(result, matrix1, matrix2, 0, R);
                return;
        }
        std::vector<std::thread> threads;
        uint8_t numThreads = std::thread::hardware_concurrency();
        threads.reserve(numThreads);
        uint64_t chunkSize = R / numThreads;
        for (uint8_t i = 0; i < numThreads; i++)
        {
                if (i == numThreads - 1)
                        threads.emplace_back(threadFunction, std::ref(result), std::ref(matrix1), std::ref(matrix2), i*chunkSize, R);
                else
                        threads.emplace_back(threadFunction, std::ref(result), std::ref(matrix1), std::ref(matrix2), i*chunkSize, (i + 1)*chunkSize);
        }
        for (uint8_t i = 0; i < numThreads; i++)
                threads[i].join();
}

// INFO: add thread is used only when R*C > THREADING_THRESHOLD
template <typename T, uint64_t R, uint64_t C>
void addThread()
{

}

// INFO: subtract thread is used only when R*C > THREADING_THRESHOLD
template <typename T, uint64_t R, uint64_t C>
void subtractThread()
{

}

template <typename T, uint64_t R, uint64_t C>
void multiplyThread(Matrix<T, R, C> &result, const Matrix<T, R, C> &matrix1, const Matrix<T, R, C> &matrix2, uint64_t start, uint64_t end)
{
        for (uint64_t i = start; i < end; i++)
        {
                for (uint64_t j = 0; j < C; j++)
                {
                        for (uint64_t k = 0; k < R; k++)
                        {
                                result(i, j) += matrix1(i, k) * matrix2(k, j);
                        }
                }
        }
}

template <typename T, uint64_t R, uint64_t C, uint64_t C2>
void multiplyThread(Matrix<T, R, C> &result, const Matrix<T, R, C> &matrix1, const Matrix<T, C, C2> &matrix2, uint64_t start, uint64_t end)
{
        for (uint64_t i = start; i < end; i++)
        {
                for (uint64_t j = 0; j < C2; j++)
                {
                        for (uint64_t k = 0; k < C; k++)
                        {
                                result(i, j) += matrix1(i, k) * matrix2(k, j);
                        }
                }
        }
}

// INFO: divide thread is used only when R*C > THREADING_THRESHOLD
template <typename T, uint64_t R, uint64_t C>
void divideThread()
{

}

// INFO: power thread is used only when R < THREADING_THRESHOLD and power > 2
template <typename T, uint64_t R, uint64_t C>
void powerThread()
{
}

// INFO: compare thread is used only when R*C > THREADING_THRESHOLD
template <typename T, uint64_t R, uint64_t C>
void compareThread()
{

}

template <typename T, uint64_t R, uint64_t C>
class Matrix<T, R, C, typename std::enable_if<!(R > BIG_MATRIX_SIZE || C > BIG_MATRIX_SIZE)>::type>
{
private:
        HeapArray<T, R*C> mMatrix;

public:
        Matrix() : mMatrix(HeapArray<T, R*C>()) {}
        Matrix(const T &value) : mMatrix(HeapArray<T, R*C>(value)) {}
        Matrix(const HeapArray<T, R*C> &matrix) : mMatrix(matrix) {}
        Matrix(const Matrix<T, R, C> &matrix) : mMatrix(matrix.mMatrix) {}

        Matrix<T, R, C> operator+(const Matrix<T, R, C> &matrix) const
        {
                Matrix<T, R, C> result;
                for (uint64_t i = 0; i < R*C; ++i)
                        result.mMatrix[i] = mMatrix[i] + matrix.mMatrix[i];
                return result;
        }

        Matrix<T, R, C> operator-(const Matrix<T, R, C> &matrix) const
        {
                Matrix<T, R, C> result;
                for (uint64_t i = 0; i < R*C; ++i)
                        result.mMatrix[i] = mMatrix[i] - matrix.mMatrix[i];
                return result;
        }

        Matrix<T, R, C> operator*(const Matrix<T, R, C> &matrix) const
        {
                Matrix<T, R, C> result;
                ThreadFunction<T, R, C> function = reinterpret_cast<ThreadFunction<T, R, C>>(multiplyThread<T, R, C>);
                dispatchThreads<T, R, C>(result, *this, matrix, function);
                return result;
        }

        template <uint64_t C2>
        Matrix<T, R, C2> operator*(const Matrix<T, C, C2> &matrix) const
        {
                Matrix<T, R, C2> result;
                // TODO: implement matrix multiplication for matrices with different sizes
                return result;
        }

        Matrix<T, R, C> operator*(const T &scalar) const
        {
                Matrix<T, R, C> result;
                for (uint64_t i = 0; i < R*C; ++i)
                        result.mMatrix[i] = mMatrix[i] * scalar;
                return result;
        }

        Matrix<T, R, C> operator/(const T &scalar) const
        {
                Matrix<T, R, C> result;
                for (uint64_t i = 0; i < R*C; ++i)
                        result.mMatrix[i] = mMatrix[i] / scalar;
                return result;
        }

        Matrix<T, R, C> &operator=(const Matrix<T, R, C> &matrix)
        {
                mMatrix = matrix.mMatrix;
                return *this;
        }

        bool operator==(const Matrix<T, R, C> &matrix) const
        {
                for (uint64_t i = 0; i < R*C; ++i)
                {
                        if (mMatrix[i] != matrix.mMatrix[i])
                                return false;
                }
                return true;
        }

        bool operator!=(const Matrix<T, R, C> &matrix) const
        {
                return !(*this == matrix);
        }

        Matrix<T, R, C> operator^(uint64_t power) const
        {
                Matrix<T, R, C> result = *this;
                for (uint64_t i = 1; i < power; i++)
                        result *= *this;
                return result;
        }

        Matrix<T, R, C> &operator^=(uint64_t power)
        {
                Matrix<T, R, C> result = *this;
                for (uint64_t i = 1; i < power; i++)
                        result *= *this;
                *this = result;
                return *this;
        }

        Matrix<T, R, C> &operator+=(const Matrix<T, R, C> &matrix)
        {
                for (uint64_t i = 0; i < R*C; ++i)
                        mMatrix[i] += matrix.mMatrix[i];
                return *this;
        }

        Matrix<T, R, C> &operator-=(const Matrix<T, R, C> &matrix)
        {
                for (uint64_t i = 0; i < R*C; ++i)
                        mMatrix[i] -= matrix.mMatrix[i];
                return *this;
        }

        Matrix<T, R, C> &operator*=(const Matrix<T, R, C> &matrix)
        {
                Matrix<T, R, C> result;
                ThreadFunction<T, R, C> function = reinterpret_cast<ThreadFunction<T, R, C>>(multiplyThread<T, R, C>);
                dispatchThreads<T, R, C>(result, *this, matrix, function);
                *this = result;
                return *this;
        }

        Matrix<T, R, C> &operator*=(const T &scalar)
        {
                for (uint64_t i = 0; i < R*C; ++i)
                        mMatrix[i] *= scalar;
                return *this;
        }

        Matrix<T, R, C> &operator/=(const T &scalar)
        {
                for (uint64_t i = 0; i < R*C; ++i)
                        mMatrix[i] /= scalar;
                return *this;
        }

        T &operator()(uint64_t row, uint64_t col)
        {
                return mMatrix[row*C + col];
        }

        const T &operator()(uint64_t row, uint64_t col) const
        {
                return mMatrix[row*C + col];
        }

        Matrix<T, R, C> &fill(const T &value)
        {
                mMatrix.fill(value);
                return *this;
        }

        Matrix<T, R, C> &zero()
        {
                mMatrix.zero();
                return *this;
        }

        Matrix<T, R, C> &diagonal(const T &value)
        {
                mMatrix.zero();
                for (uint64_t i = 0; i < R; ++i)
                        mMatrix[i*C + i] = value;
                return *this;
        }

        Matrix<T, R, C> &identity()
        {
                mMatrix.zero();
                for (uint64_t i = 0; i < R; ++i)
                        mMatrix[i*C + i] = 1;
                return *this;
        }

        Matrix<T, C, R> transpose() const
        {
                Matrix<T, C, R> result;
                for (uint64_t i = 0; i < R; ++i)
                {
                        for (uint64_t j = 0; j < C; ++j)
                                result.mMatrix[j*R + i] = mMatrix[i*C + j];
                }
                return result;
        }

        // TODO: implement inverseInPlace algorithm
        void inverseInPlace();

        Matrix<T, R, C> inverse() const
        {
                Matrix<T, R, C> result = *this;
                result.inverseInPlace();
                return result;
        }

        void print() const
        {
                for (uint64_t i = 0; i < R; ++i)
                {
                        for (uint64_t j = 0; j < C; ++j)
                                std::cout << mMatrix[i*C + j] << ((j + 1 < C) ? " " : "");
                        std::cout << "\n";
                }
        }

        T trace() const
        {
                if (R != C)
                        throw std::runtime_error("Matrix must be square to calculate trace.");
                T result = 0;
                for (uint64_t i = 0; i < R; ++i)
                        result += mMatrix[i*C + i];
                return result;
        }

        // TODO: implement determinant calculation for matrices larger than 3x3
        T determinant() const
        {
                if (R != C)
                        throw std::runtime_error("Matrix must be square to calculate determinant.");
                if (R == 1)
                        return mMatrix[0];
                if (R == 2)
                        return mMatrix[0]*mMatrix[3] - mMatrix[1]*mMatrix[2];
                if (R == 3)
                        return mMatrix[0]*(mMatrix[4]*mMatrix[8] - mMatrix[5]*mMatrix[7]) -
                               mMatrix[1]*(mMatrix[3]*mMatrix[8] - mMatrix[5]*mMatrix[6]) +
                               mMatrix[2]*(mMatrix[3]*mMatrix[7] - mMatrix[4]*mMatrix[6]);
                throw std::runtime_error("Determinant calculation not implemented for matrices larger than 3x3.");
        }
};

template <typename T, uint64_t R, uint64_t C>
class Matrix<T, R, C, typename std::enable_if<(R > BIG_MATRIX_SIZE || C > BIG_MATRIX_SIZE)>::type>
{
public:
        Matrix<T, R, C> operator*(const Matrix<T, R, C> &matrix) const;
        Matrix<T, R, C> &operator*=(const Matrix<T, R, C> &matrix);
};

#endif // MATRIX_HPP
