#include "Model.hh"
#include <iostream>
#include <chrono>
#include <iomanip>

void Model::add(const Layer& layer) {
    layers.push_back(layer);
}

std::vector<Layer>& Model::getLayers(){
    return this->layers;
}

Tensor<double> Model::forward(Tensor<double> x) {
    Tensor<double> current = x;
    for (Layer& layer : layers) {
        current = layer.forward(current);
    }
    return current;
}

void Model::fit(const std::vector<std::pair<Tensor<double>, double>>& dataset, size_t epochs) {
    std::cout << "Starting training on " << dataset.size() << " samples...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        
        for (const auto& data_point : dataset) {
            const Tensor<double>& x = data_point.first;
            double target_y = data_point.second;
            
            // Forward pass through the whole network
            Tensor<double> prediction_tensor = this->forward(x);
            double prediction = prediction_tensor.getIndexMem(0);
            
            // Calculate Error (MSE)
            double error = prediction - target_y;
            total_loss += 0.5 * error * error;
            
            // Initial Gradient for Output Layer
            Tensor<double> gradient({1});
            gradient.setIndexMem(0, error);
            
            // Backward pass through the network (in reverse)
            for (int i = layers.size() - 1; i >= 0; --i) {
                gradient = layers[i].backward(gradient);
            }
        }
        
        if (epoch % 10 == 0 || epoch == epochs - 1) {
            double avg_loss = total_loss / dataset.size();
            std::cout << "Epoch " << epoch << " | Average Loss: " << avg_loss << "\n";
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> training_time = end_time - start_time;
    std::cout << "Training complete in " << training_time.count() << " seconds!\n\n";
}

void Model::evaluate(const std::vector<std::pair<Tensor<double>, double>>& dataset) {
    std::cout << "================ TESTING PHASE ================\n";
    double test_loss = 0.0;
    
    std::cout << std::left << std::setw(15) << "Prediction" 
              << std::left << std::setw(15) << "Target" 
              << "Error\n";
    std::cout << "-----------------------------------------------\n";
    
    for (size_t i = 0; i < dataset.size(); ++i) {
        const Tensor<double>& x = dataset[i].first;
        double target_y = dataset[i].second;
        
        Tensor<double> prediction_tensor = this->forward(x);
        double prediction = prediction_tensor.getIndexMem(0);
        
        double error = prediction - target_y;
        test_loss += 0.5 * error * error;
        
        if (i < 10) {
            std::cout << std::left << std::setw(15) << prediction 
                      << std::left << std::setw(15) << target_y 
                      << error << "\n";
        }
    }
    
    double avg_test_loss = test_loss / dataset.size();
    std::cout << "\nFinal Test MSE: " << avg_test_loss << "\n";
    std::cout << "===============================================\n";
}
