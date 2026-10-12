#pragma once
#include <vector>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

// Struct holding aggregated evaluation metrics
struct EvaluationResult {
    double loss = 0.0;
    double mse = 0.0;
    double rmse = 0.0;
    double mae = 0.0;
    double accuracy = 0.0;
    size_t sample_count = 0;

    void print(const std::string& split_name = "Test", const std::string& loss_name = "Loss") const {
        std::cout << "\n================ " << split_name << " Evaluation Results ================\n";
        std::cout << "  Samples Evaluated:        " << sample_count << "\n";
        std::cout << "  " << std::left << std::setw(26) << (loss_name + " Loss:") << std::defaultfloat << loss << "\n";
        std::cout << "  Mean Squared Error (MSE): " << std::defaultfloat << mse << "\n";
        std::cout << "  Root MSE (RMSE):          " << std::defaultfloat << rmse << "\n";
        std::cout << "  Mean Absolute Error (MAE):" << std::defaultfloat << mae << "\n";
        std::cout << "  Classification Accuracy:  " << std::fixed << std::setprecision(2) << (accuracy * 100.0) << "%\n";
        std::cout << std::defaultfloat;
        std::cout << "========================================================\n\n";
    }
};

// Error and Metric calculations
class Metrics {
public:
    // Mean Squared Error: sum((y_pred - y_true)^2) / N
    static double mse(const std::vector<double>& preds, const std::vector<double>& targets) {
        if (preds.empty()) return 0.0;
        double sum = 0.0;
        for (size_t i = 0; i < preds.size(); ++i) {
            double diff = preds[i] - targets[i];
            sum += diff * diff;
        }
        return sum / preds.size();
    }

    // Root Mean Squared Error
    static double rmse(const std::vector<double>& preds, const std::vector<double>& targets) {
        return std::sqrt(mse(preds, targets));
    }

    // Mean Absolute Error: sum(|y_pred - y_true|) / N
    static double mae(const std::vector<double>& preds, const std::vector<double>& targets) {
        if (preds.empty()) return 0.0;
        double sum = 0.0;
        for (size_t i = 0; i < preds.size(); ++i) {
            sum += std::abs(preds[i] - targets[i]);
        }
        return sum / preds.size();
    }

    // Classification Accuracy: percentage of predictions matching target (using threshold, default 0.5)
    static double accuracy(const std::vector<double>& preds, const std::vector<double>& targets, double threshold = 0.5) {
        if (preds.empty()) return 0.0;
        size_t correct = 0;
        for (size_t i = 0; i < preds.size(); ++i) {
            double pred_class = (preds[i] >= threshold) ? 1.0 : 0.0;
            double target_class = (targets[i] >= threshold) ? 1.0 : 0.0;
            if (pred_class == target_class) {
                correct++;
            }
        }
        return static_cast<double>(correct) / preds.size();
    }
};
