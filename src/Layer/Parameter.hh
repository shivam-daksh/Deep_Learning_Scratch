#pragma once
#include "Tensor.hh"

template <typename DType>
class Parameter {
    public:
        Tensor<DType> value;
        Tensor<DType> gradient;
        bool regularize = true;

        Parameter() = default;
        Parameter(const std::vector<size_t> &shape, bool regularize = true) : regularize(regularize) {
            value.resize(shape);
            gradient.resize(shape);
        }
};