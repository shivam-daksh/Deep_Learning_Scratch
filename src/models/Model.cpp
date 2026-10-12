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

void Model::fit(const std::vector<std::pair<Tensor<double>, double>>& dataset, 
                size_t epochs, 
                size_t batch_size, 
                Optimizer* optimizer, 
                Loss* loss_fn,
                const std::vector<std::pair<Tensor<double>, double>>* val_dataset) {
    MSELoss default_loss;
    Loss* active_loss = loss_fn ? loss_fn : &default_loss;

    std::cout << "Starting Mini-Batch training on " << dataset.size() << " samples with Batch Size " << batch_size 
              << " (" << active_loss->name() << " Loss)";
    if (val_dataset && !val_dataset->empty()) {
        std::cout << " [Validation: " << val_dataset->size() << " samples]";
    }
    std::cout << "...\n";
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
                
                double sample_loss = active_loss->calculate(prediction, target_y);
                total_loss += sample_loss;
                
                grad_ptr[b] = active_loss->gradient(prediction, target_y) / current_batch_size;
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
            std::cout << "Epoch " << std::setw(3) << epoch << " | Train " << active_loss->name() << ": " << std::defaultfloat << avg_loss;
            if (val_dataset && !val_dataset->empty()) {
                EvaluationResult val_res = this->evaluate(*val_dataset, active_loss, false);
                std::cout << " | Val " << active_loss->name() << ": " << std::defaultfloat << val_res.loss
                          << " | Val Acc: " << std::fixed << std::setprecision(2) << (val_res.accuracy * 100.0) << "%"
                          << std::defaultfloat;
            }
            std::cout << "\n";
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> training_time = end_time - start_time;
    std::cout << "Training complete in " << training_time.count() << " seconds!\n\n";
}

EvaluationResult Model::evaluate(const std::vector<std::pair<Tensor<double>, double>>& dataset, Loss* loss_fn, bool verbose) {
    MSELoss default_loss;
    Loss* active_loss = loss_fn ? loss_fn : &default_loss;

    if (dataset.empty()) {
        return EvaluationResult{};
    }

    std::vector<double> preds(dataset.size());
    std::vector<double> targets(dataset.size());
    double total_loss = 0.0;

    if (verbose) {
        std::cout << "================ TESTING PHASE (" << active_loss->name() << ") ================\n";
        std::cout << std::left << std::setw(15) << "Prediction" 
                  << std::left << std::setw(15) << "Target" 
                  << "Residual\n";
        std::cout << "-----------------------------------------------\n";
    }

    for (size_t i = 0; i < dataset.size(); ++i) {
        const Tensor<double>& x = dataset[i].first;
        double target_y = dataset[i].second;
        
        Tensor<double> prediction_tensor = this->forward(x);
        double prediction = prediction_tensor.getIndexMem(0);
        preds[i] = prediction;
        targets[i] = target_y;

        double sample_loss = active_loss->calculate(prediction, target_y);
        total_loss += sample_loss;

        if (verbose && i < 10) {
            std::cout << std::left << std::setw(15) << std::defaultfloat << prediction 
                      << std::left << std::setw(15) << std::defaultfloat << target_y 
                      << std::defaultfloat << (prediction - target_y) << "\n";
        }
    }

    EvaluationResult res;
    res.sample_count = dataset.size();
    res.loss = total_loss / dataset.size();
    res.mse = Metrics::mse(preds, targets);
    res.rmse = Metrics::rmse(preds, targets);
    res.mae = Metrics::mae(preds, targets);
    res.accuracy = Metrics::accuracy(preds, targets);

    if (verbose) {
        std::cout << "-----------------------------------------------\n";
        res.print("Evaluation Summary", active_loss->name());
    }

    return res;
}
