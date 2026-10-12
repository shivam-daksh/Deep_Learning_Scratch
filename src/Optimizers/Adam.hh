#pragma once
#include <unordered_map>
#include "Optimizer.hh"
#include "Model.hh"

class Adam : public Optimizer{
    private:
        Model &model;
        double learningRate, beta1, beta2;
        std::unordered_map<Parameter<double>*, std::pair<Tensor<double>, Tensor<double>>> velocities;
        // std::vector<Tensor<double>> vel_w, vel_b, m_w, m_b;

    public:
        Adam(Model &model, const double beta1, const double beta2, const double lr, double weight_decay = 0.0, double l1 = 0.0);
        void step() override;
        ~Adam() override = default;
};