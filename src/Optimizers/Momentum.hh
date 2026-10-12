#pragma once
#include <unordered_map>
#include "Optimizer.hh"
#include "Tensor.hh"


class Momentum : public Optimizer {
    private:
        Model &model;
        double learningRate;
        double beta;

        std::unordered_map<Parameter<double>*, Tensor<double>> velocities;
        // std::vector<Tensor<double>> vel_w, vel_b;
    public:
        explicit Momentum(Model &model, const double beta, const double lr, double weight_decay = 0.0, double l1 = 0.0);
        void step() override;
        ~Momentum() override = default;
};