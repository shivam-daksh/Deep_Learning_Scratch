// part of this code is AI-Generated

#pragma once
#include <vector>
#include <utility>
#include <algorithm>
#include <random>
#include "Tensor.hh"

// Holds partitioned dataset splits: train, validation, and test
struct DatasetSplit {
    std::vector<std::pair<Tensor<double>, double>> train;
    std::vector<std::pair<Tensor<double>, double>> val;
    std::vector<std::pair<Tensor<double>, double>> test;
};

// Splits a dataset into train, validation, and test subsets
inline DatasetSplit train_test_split(
    const std::vector<std::pair<Tensor<double>, double>>& data,
    double train_ratio = 0.70,
    double val_ratio = 0.15,
    double test_ratio = 0.15,
    bool shuffle = true,
    unsigned int seed = 42
) {
    std::vector<std::pair<Tensor<double>, double>> copy_data = data;
    if (shuffle) {
        std::mt19937 g(seed);
        std::shuffle(copy_data.begin(), copy_data.end(), g);
    }

    size_t total = copy_data.size();
    size_t train_end = static_cast<size_t>(total * train_ratio);
    size_t val_end = train_end + static_cast<size_t>(total * val_ratio);

    DatasetSplit split;
    split.train.assign(copy_data.begin(), copy_data.begin() + train_end);
    split.val.assign(copy_data.begin() + train_end, copy_data.begin() + val_end);
    split.test.assign(copy_data.begin() + val_end, copy_data.end());

    return split;
}

