#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <cstdint>

#include "heaparray.hpp"

template <typename T, uint64_t R, uint64_t C, uint64_t D>
class Tensor
{
private:
        HeapArray<T, R*C*D> mTensor;

public:
        Tensor() : mTensor(Vector<T, R*C*D>()) {}
        Tensor(const HeapArray<T, R*C*D>& tensor) : mTensor(tensor) {}
        Tensor(const Tensor<T, R, C, D>& other) : mTensor(other.mTensor) {}
};

#endif // TENSOR_HPP
