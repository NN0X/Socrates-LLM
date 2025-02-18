#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <iostream>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <cassert>

#include "heaparray.hpp"

template <typename T, uint64_t N>
class Vector
{
private:
        HeapArray<T, N> mVector;

public:
        Vector() : mVector(HeapArray<T, N>()) {}
        Vector(const HeapArray<T, N>& vector) : mVector(vector) {}
        Vector(const Vector<T, N>& other) : mVector(other.mVector) {}
        Vector(const T& value) : mVector(HeapArray<T, N>(value)) {}

        template <typename... Args>
        Vector(Args... args) : mVector(std::array<T, N>{static_cast<T>(args)...})
        {
                static_assert(sizeof...(args) == N, "Number of arguments must match the size of the vector.");
        }

        Vector<T, N> operator+(const Vector<T, N>& other) const
        {
                Vector<T, N> result;
                for (uint64_t i = 0; i < N; ++i)
                        result.mVector[i] = mVector[i] + other.mVector[i];
                return result;
        }

        Vector<T, N> operator-(const Vector<T, N>& other) const
        {
                Vector<T, N> result;
                for (uint64_t i = 0; i < N; ++i)
                        result.mVector[i] = mVector[i] - other.mVector[i];
                return result;
        }

        Vector<T, N> operator*(const T& scalar) const
        {
                Vector<T, N> result;
                for (uint64_t i = 0; i < N; ++i)
                        result.mVector[i] = mVector[i] * scalar;
                return result;
        }

        Vector<T, N> operator/(const T& scalar) const
        {
                Vector<T, N> result;
                for (uint64_t i = 0; i < N; ++i)
                        result.mVector[i] = mVector[i] / scalar;
                return result;
        }

        Vector<T, N>& operator=(const Vector<T, N>& other)
        {
                mVector = other.mVector;
                return *this;
        }

        bool operator==(const Vector<T, N>& other) const
        {
                for (uint64_t i = 0; i < N; ++i)
                {
                        if (mVector[i] != other.mVector[i])
                                return false;
                }
                return true;
        }

        bool operator!=(const Vector<T, N>& other) const
        {
                return !(*this == other);
        }

        Vector<T, N> operator^(uint64_t power) const
        {
                Vector<T, N> result = *this;
                for (uint64_t i = 1; i < power; i++)
                        result *= *this;
                return result;
        }

        Vector<T, N>& operator^=(uint64_t power)
        {
                Vector<T, N> result = *this;
                for (uint64_t i = 1; i < power; i++)
                        result *= *this;
                *this = result;
                return *this;
        }

        Vector<T, N>& operator+=(const Vector<T, N>& other)
        {
                for (uint64_t i = 0; i < N; ++i)
                        mVector[i] += other.mVector[i];
                return *this;
        }

        Vector<T, N>& operator-=(const Vector<T, N>& other)
        {
                for (uint64_t i = 0; i < N; ++i)
                        mVector[i] -= other.mVector[i];
                return *this;
        }

        Vector<T, N>& operator*=(const T& scalar)
        {
                for (uint64_t i = 0; i < N; ++i)
                        mVector[i] *= scalar;
                return *this;
        }

        Vector<T, N>& operator/=(const T& scalar)
        {
                for (uint64_t i = 0; i < N; ++i)
                        mVector[i] /= scalar;
                return *this;
        }

        T& operator[](uint64_t index)
        {
                return mVector[index];
        }

        const T& operator[](uint64_t index) const
        {
                return mVector[index];
        }

        Vector<T, N>& fill(const T& value)
        {
                mVector.fill(value);
                return *this;
        }

        Vector<T, N>& zero()
        {
                mVector.zero();
                return *this;
        }

        T dot(const Vector<T, N>& other) const
        {
                T result = 0;
                for (uint64_t i = 0; i < N; ++i)
                        result += mVector[i] * other.mVector[i];
                return result;
        }

        T magnitude() const
        {
                return sqrt(dot(*this));
        }

        Vector<T, N> normalize() const
        {
                return *this / magnitude();
        }

        Vector<T, N> cross(const Vector<T, N>& other) const
        {
                Vector<T, N> result;
                for (uint64_t i = 0; i < N; ++i)
                        result[i] = mVector[(i + 1) % N] * other.mVector[(i + 2) % N] - mVector[(i + 2) % N] * other.mVector[(i + 1) % N];
                return result;
        }

        void print() const
        {
                for (uint64_t i = 0; i < N; ++i)
                        std::cout << mVector[i] << " ";
                std::cout << "\n";
        }
};

#endif // VECTOR_HPP
