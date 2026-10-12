#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "Model.hh"
#include "Tensor.hh"
#include "SGD.hh"
#include "Momentum.hh"
#include "Adam.hh"
#include "Loss/Loss.hh"
#include "Metrics/Metrics.hh"
#include "utils/Dataset.hh"

int main() {
    std::string filename = "big_dataset.csv";
    std::ifstream file(filename);

    if(!file.is_open()){
        file.open("../" + filename);
    }

    if(!file.is_open()){
        std::cerr << "Failed to open " << filename << ".\n";
        return 1;
    }

    std::vector<std::pair<Tensor<double>, double>> dataset;
    std::string line;
    std::getline(file, line); // skip header

    size_t numCols = 1;
    for(char c : line) { if(c == ',') numCols++; }
    size_t inputFeatures = numCols - 1;

    while(std::getline(file, line)){
        if(line.empty()) continue;
        std::stringstream ss(line);
        std::string val;

        std::vector<double> row_vals;
        while(std::getline(ss, val, ',')){
            row_vals.push_back(std::stod(val));
        }

        if(row_vals.size() != numCols) continue;

        Tensor<double> input({inputFeatures});
        for(size_t i = 0; i < inputFeatures; i++){
            input({i}) = row_vals[i];
        }

        double y = row_vals.back();
        dataset.push_back({input, y});
    }

    // --- TRAIN / VALIDATION / TEST SPLIT (70% / 15% / 15%) ---
    DatasetSplit split = train_test_split(dataset, 0.70, 0.15, 0.15, true, 42);
    std::cout << "Dataset Split:\n"
              << "  Train samples: " << split.train.size() << "\n"
              << "  Val   samples: " << split.val.size() << "\n"
              << "  Test  samples: " << split.test.size() << "\n\n";

    // --- BUILD THE MODEL ---
    Model model;
    
    // Hidden Layer
    Layer hidden_layer(8, inputFeatures, InitType::HE);
    hidden_layer.setActivation(Activation::RELU);
    model.add(hidden_layer);
    
    // Output Layer
    Layer output_layer(1, 8, InitType::XAVIER);
    output_layer.setActivation(Activation::SIGMOID);
    model.add(output_layer);
    
    // Loss Function: Binary Cross Entropy
    BCELoss bce_loss;

    // Optimizer: Adam with L2 Weight Decay (Regularization)
    double learning_rate = 0.01;
    double weight_decay = 1e-4; // L2 Regularization penalty
    Adam optimizer(model, 0.9, 0.999, learning_rate, weight_decay);

    // Train on train set with validation monitoring
    model.fit(split.train, 100, 32, &optimizer, &bce_loss, &split.val);

    // Final Evaluation on Unseen Test Set
    model.evaluate(split.test, &bce_loss, true);

    return 0;
}
