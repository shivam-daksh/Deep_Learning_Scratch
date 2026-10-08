#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "Model.hh"
#include "Tensor.hh"
#include "SGD.hh"

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

    // --- BUILD THE MODEL ---
    Model model;
    
    // Hidden Layer
    Layer hidden_layer(8, inputFeatures);
    hidden_layer.setLearningRate(0.01);
    hidden_layer.setActivation(Activation::RELU);
    model.add(hidden_layer);
    
    // Output Layer
    Layer output_layer(1, 8);
    output_layer.setLearningRate(0.01);
    output_layer.setActivation(Activation::SIGMOID);
    model.add(output_layer);
    
    // Train and Evaluate
    SGD optimizer(0.01);
    model.fit(dataset, 40, &optimizer);
    model.evaluate(dataset);

    return 0;
}
