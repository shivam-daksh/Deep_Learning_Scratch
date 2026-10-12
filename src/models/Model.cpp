#include <iostream>
#include <chrono>
#include <iomanip>
#include "Model.hh"
#include "Optimizer.hh"

Model::Model(const size_t batch_size) : batch_size(batch_size){}

void Model::add(const Layer& layer) {
    layers.push_back(std::make_unique<Layer>(layer));
}

std::vector<std::unique_ptr<Layer>>& Model::getLayers(){
    return this->layers;
}

Tensor<double> Model::forward(const Tensor<double>& x) {
    Tensor<double> current = x;
    for (std::unique_ptr<Layer>& layer : layers) {
        current = layer->forward(current);
    }
    return current;
}

void Model::fit(const std::vector<std::pair<Tensor<double>, double>>& dataset, size_t epochs, size_t batch_size, Optimizer* optimizer) {
    std::cout << "Starting Mini-Batch training on " << dataset.size() << " samples with Batch Size " << batch_size << "...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        
        // batch training
        for (size_t batch_start = 0; batch_start < dataset.size(); batch_start += batch_size) {
            size_t current_batch_size = std::min(batch_size, dataset.size() - batch_start);
            
            // segregate the input and target features
            size_t inputFeatures = dataset[0].first.getSize();
            Tensor<double> X_batch({current_batch_size, inputFeatures});
            Tensor<double> Y_batch({current_batch_size, 1});
            double* y_ptr = Y_batch.data();
            double* x_batch_ptr = X_batch.data();
            
            for (size_t b = 0; b < current_batch_size; ++b) {
                const Tensor<double>& x = dataset[batch_start + b].first;
                y_ptr[b] = dataset[batch_start + b].second;
                const double* x_src = x.data();
                for (size_t i = 0; i < inputFeatures; ++i) {
                    x_batch_ptr[b * inputFeatures + i] = x_src[i];
                }
            }
            
            // forward / prediction
            Tensor<double> prediction_tensor = this->forward(X_batch);
            
            // calculate the gradient
            Tensor<double> gradient({current_batch_size, 1});
            double* grad_ptr = gradient.data();
            const double* pred_ptr = prediction_tensor.data();

            for (size_t b = 0; b < current_batch_size; ++b) {
                double prediction = pred_ptr[b];
                double target_y = y_ptr[b];
                
                double error = prediction - target_y;
                total_loss += 0.5 * error * error;
                
                grad_ptr[b] = error / current_batch_size;
            }
            
            // backward / calculate the gradient of all layers
            for (int i = layers.size() - 1; i >= 0; --i) {
                gradient = layers[i]->backward(gradient);
            }
            
            // update the weights
            if (optimizer) {
                optimizer->step();
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
