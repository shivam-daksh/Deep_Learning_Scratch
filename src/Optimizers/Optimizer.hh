#pragma once
#include "Tensor.hh"
#include "Model.hh"
// Interface for optimizer
class Optimizer {
    public:
        virtual void step() = 0;
        virtual ~Optimizer() = default;
};