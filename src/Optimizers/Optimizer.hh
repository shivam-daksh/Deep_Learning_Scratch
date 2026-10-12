#pragma once
#include "Tensor.hh"
#include "Model.hh"
// Interface for optimizer
class Optimizer {
    protected:
        double weight_decay = 0.0; // L2 Regularization (Weight Decay)
        double l1_ratio = 0.0;     // L1 Regularization (Lasso)

        inline double applyRegularization(double grad, double param_val, bool regularize = true) const {
            if (!regularize) {
                return grad;
            }
            double reg_grad = grad;
            if (this->weight_decay > 0.0) {
                reg_grad += this->weight_decay * param_val;
            }
            if (this->l1_ratio > 0.0) {
                reg_grad += this->l1_ratio * (param_val > 0.0 ? 1.0 : (param_val < 0.0 ? -1.0 : 0.0));
            }
            return reg_grad;
        }

    public:
        virtual void step() = 0;
        virtual ~Optimizer() = default;

        void setWeightDecay(double l2) { this->weight_decay = l2; }
        void setL1(double l1) { this->l1_ratio = l1; }
        void setRegularization(double l2, double l1 = 0.0) {
            this->weight_decay = l2;
            this->l1_ratio = l1;
        }
        double getWeightDecay() const { return this->weight_decay; }
        double getL1() const { return this->l1_ratio; }
};