#ifndef HEAPARRAY_HPP
#define HEAPARRAY_HPP

#include <iostream>
#include <cstdint>
#include <array>
#include <memory>

template <typename T, uint64_t N>
class HeapArray
{
private:
        std::unique_ptr<std::array<T, N>> mArray;

public:
        HeapArray() : mArray(std::make_unique<std::array<T, N>>())
        {
                mArray->fill(0);
        };

        HeapArray(const T &value) : mArray(std::make_unique<std::array<T, N>>())
        {
                mArray->fill(value);

        }

        HeapArray(const std::array<T, N> &array) : mArray(std::make_unique<std::array<T, N>>())
        {
                *mArray = array;
        }

        HeapArray(const HeapArray<T, N> &array) : mArray(std::make_unique<std::array<T, N>>())
        {
                *mArray = *array.mArray;
        }

        HeapArray<T, N> &operator=(const HeapArray<T, N> &array)
        {
                *mArray = *array.mArray;
                return *this;
        }

        T &operator[](uint64_t index)
        {
                return (*mArray)[index];
        }

        const T &operator[](uint64_t index) const
        {
                return (*mArray)[index];
        }

        void fill(const T &value)
        {
                mArray->fill(value);
        }

        void zero()
        {
                mArray->fill(0);
        }

        void print() const
        {
                for (uint64_t i = 0; i < N; ++i)
                        std::cout << (*mArray)[i] << ((i + 1 < N) ? " " : "");
                std::cout << "\n";
        }

        uint64_t size() const
        {
                return N;
        }
};

#endif // HEAPARRAY_HPP
