#pragma once
#include <vector>
#include <memory>
#include "Layer.hh"


class Optimizer;

class Model {
    private:
        size_t batch_size = 32;
        std::vector<std::unique_ptr<Layer>> layers;
        
    public:
        Model() = default;
        Model(const size_t batch_size);
        void add(const Layer& layer);
        std::vector<std::unique_ptr<Layer>>& getLayers();
        Tensor<double> forward(const Tensor<double>& x);
        void fit(const std::vector<std::pair<Tensor<double>, double>>& dataset, size_t epochs, size_t batch_size = 32, Optimizer* optimizer = nullptr);
        void evaluate(const std::vector<std::pair<Tensor<double>, double>>& dataset);
};
