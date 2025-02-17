// TODO: use Strassen Algorithm for big matrix multiplication
// TODO: check the matrix sizes at which Strassen Algorithm is faster than normal multiplication

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <array>
#include <cstdint>
#include <thread>
#include <vector>
#include <type_traits>

#define BIG_MATRIX_SIZE 64

template <typename T, uint64_t R, uint64_t C, typename Enable = void>
class Matrix;

template <typename T, uint64_t R, uint64_t C>
class Matrix<T, R, C, typename std::enable_if<!(R > BIG_MATRIX_SIZE || C > BIG_MATRIX_SIZE)>::type>
{
private:
        std::array<T, R*C> mMatrix;
        std::vector<std::thread> mThreads;
        const uint8_t mNumThreads = std::thread::hardware_concurrency();

public:
        Matrix()
        {
                mMatrix.fill(0);
                mThreads.reserve(mNumThreads);
        }

        Matrix(const T &value)
        {
                mMatrix.fill(value);
                mThreads.reserve(mNumThreads);
        }

        Matrix(const std::array<T, R*C> &matrix) : mMatrix(matrix) 
        {
                mThreads.reserve(mNumThreads);
        }
        Matrix(const Matrix<T, R, C> &matrix) : mMatrix(matrix.mMatrix) 
        {
                mThreads.reserve(mNumThreads);

        }

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
                for (uint64_t i = 0; i < R; ++i)
                {
                        for (uint64_t j = 0; j < C; ++j)
                        {
                                for (uint64_t k = 0; k < R; ++k)
                                        result.mMatrix[i*C + j] += mMatrix[i*C + k] * matrix.mMatrix[k*C + j];
                        }
                }
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
                for (uint64_t i = 1; i < power; ++i)
                        result *= *this;
                return result;
        }

        Matrix<T, R, C> &operator^=(uint64_t power)
        {
                Matrix<T, R, C> result = *this;
                for (uint64_t i = 1; i < power; ++i)
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
                for (uint64_t i = 0; i < R; ++i)
                {
                        for (uint64_t j = 0; j < C; ++j)
                        {
                                for (uint64_t k = 0; k < R; ++k)
                                        result.mMatrix[i*C + j] += mMatrix[i*C + k] * matrix.mMatrix[k*C + j];
                        }
                }
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
                mMatrix.fill(0);
                return *this;
        }

        Matrix<T, R, C> &identity()
        {
                mMatrix.fill(0);
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

        Matrix<T, R, C> inverse() const
        {
                Matrix<T, R, C> result = *this;
                result.inverseInPlace();
                return result;
        }

        void inverseInPlace();

        void print() const
        {
                for (uint64_t i = 0; i < R; ++i)
                {
                        for (uint64_t j = 0; j < C; ++j)
                                std::cout << mMatrix[i*C + j] << ((j + 1 < C) ? " " : "");
                        std::cout << "\n";
                }
        }

        T determinant() const;
        T trace() const;
};

template <typename T, uint64_t R, uint64_t C>
class Matrix<T, R, C, typename std::enable_if<(R > BIG_MATRIX_SIZE || C > BIG_MATRIX_SIZE)>::type>
{
public:
        Matrix<T, R, C> operator*(const Matrix<T, R, C> &matrix) const;
        Matrix<T, R, C> &operator*=(const Matrix<T, R, C> &matrix);
};

#endif // MATRIX_HPP
