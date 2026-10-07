#pragma once
#include "Layer.hh"
#include <vector>

class Model {
    private:
        std::vector<Layer> layers;
        
    public:
        Model() = default;
        void add(const Layer& layer);
        std::vector<Layer>& getLayers();
        Tensor<double> forward(Tensor<double> x);
        void fit(const std::vector<std::pair<Tensor<double>, double>>& dataset, size_t epochs);
        void evaluate(const std::vector<std::pair<Tensor<double>, double>>& dataset);
};
