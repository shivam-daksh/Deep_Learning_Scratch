//
#pragma once
#include <cmath>
#include <algorithm>
#include <string>

class Loss {
public:
    virtual ~Loss() = default;
    virtual double calculate(double pred, double target) const = 0;
    virtual double gradient(double pred, double target) const = 0;
    virtual std::string name() const = 0;
};

// Mean Squared Error (MSE)
class MSELoss : public Loss {
public:
    double calculate(double pred, double target) const override {
        double diff = pred - target;
        return 0.5 * diff * diff;
    }

    double gradient(double pred, double target) const override {
        return pred - target;
    }

    std::string name() const override {
        return "MSE";
    }
};

// Mean Absolute Error 
class MAELoss : public Loss {
public:
    double calculate(double pred, double target) const override {
        return std::abs(pred - target);
    }

    double gradient(double pred, double target) const override {
        double diff = pred - target;
        if (diff > 0.0) return 1.0;
        if (diff < 0.0) return -1.0;
        return 0.0;
    }

    std::string name() const override {
        return "MAE";
    }
};

// Huber Loss (Smooth L1 Loss)
class HuberLoss : public Loss {
private:
    double delta;
public:
    explicit HuberLoss(double delta = 1.0) : delta(delta) {}

    double calculate(double pred, double target) const override {
        double diff = std::abs(pred - target);
        if (diff <= delta) {
            return 0.5 * diff * diff;
        } else {
            return delta * (diff - 0.5 * delta);
        }
    }

    double gradient(double pred, double target) const override {
        double diff = pred - target;
        if (std::abs(diff) <= delta) {
            return diff;
        } else {
            return delta * (diff > 0.0 ? 1.0 : -1.0);
        }
    }

    std::string name() const override {
        return "Huber";
    }
};

// Binary Cross Entropy (BCE)
class BCELoss : public Loss {
public:
    double calculate(double pred, double target) const override {
        const double eps = 1e-15;
        double p = std::clamp(pred, eps, 1.0 - eps);
        return -(target * std::log(p) + (1.0 - target) * std::log(1.0 - p));
    }

    double gradient(double pred, double target) const override {
        const double eps = 1e-15;
        double p = std::clamp(pred, eps, 1.0 - eps);
        return (p - target) / (p * (1.0 - p));
    }

    std::string name() const override {
        return "BCE";
    }
};
