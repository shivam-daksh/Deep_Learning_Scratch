#pragma once
#include "Optimizer.hh"

class SGD : public Optimizer {
    private:
        double learningRate;
        Model &model;
    
    public:
        explicit SGD(Model &model, double lr, double weight_decay = 0.0, double l1 = 0.0);
        void step() override;

        ~SGD() override = default;
};