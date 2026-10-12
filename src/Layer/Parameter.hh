#pragma once
#include "Tensor.hh"

template <typename DType>
class Parameter {
    public:
        Tensor<DType> value;
        Tensor<DType> gradient;
        Parameter() = default;
        Parameter(const std::vector<size_t> &shape){
            value.resize(shape);
            gradient.resize(shape);
        }
};