#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include "Tensor.hh"


enum class Activation { SIGMOID, RELU, LINEAR };

class Layer {
    private:
        Tensor<double> weights;
        Tensor<double> bias;
        Activation activationType = Activation::LINEAR;
        double learningRate = 0.01;
        size_t epochs = 50;
        
        // Cache the output for the backward pass
        Tensor<double> last_output;

    public:
        Layer(size_t no_of_neurons, size_t inputSize) : weights({no_of_neurons, inputSize}), bias({no_of_neurons}) {
            // Random weight initialization (CRITICAL to break symmetry in Deep Networks!)
            for(size_t i = 0; i < weights.getSize(); i++) {
                double r = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // Random between -1 and 1
                weights.setIndexMem(i, r);
            }
            for(size_t i = 0; i < bias.getSize(); i++) {
                bias.setIndexMem(i, 0.0);
            }
        }
        
        void setEpochs(size_t epochs){
            this->epochs = epochs;
        }

        void setLearningRate(double lr){
            this->learningRate = lr;
        }
        
        void setActivation(Activation type) {
            this->activationType = type;
        }

        Tensor<double> forward(const Tensor<double>&input){
            Tensor<double> res = this->weights.matmul(input);
            res = res + this->bias;
            
            // Apply Activation Function
            for(size_t i = 0; i < res.getSize(); i++){
                double val = res.getIndexMem(i);
                if (activationType == Activation::SIGMOID) {
                    res.setIndexMem(i, 1.0 / (1.0 + std::exp(-val)));
                } else if (activationType == Activation::RELU) {
                    res.setIndexMem(i, val > 0 ? val : 0.0);
                }
                // If LINEAR, do nothing (f(x) = x)
            }
            
            this->last_output = res;
            return res;
        }

        Tensor<double> backward(const Tensor<double>& input, const Tensor<double>& dL_dout){
            size_t inputSize = input.getSize();
            size_t numNeurons = this->bias.getSize();
            
            // Create a Tensor to hold the error we will pass back to the previous layer
            Tensor<double> dL_dInput(input.getShape());
            for(size_t i = 0; i < inputSize; i++) {
                dL_dInput.setIndexMem(i, 0.0); // Initialize with zeros
            }

            // Iterate over all neurons in this layer
            for (size_t n = 0; n < numNeurons; n++) {
                double output = this->last_output.getIndexMem(n);
                
                // Loss Derivative now comes from the parameter (passed from the layer ahead!)
                double current_dL_dout = dL_dout.getIndexMem(n);
                
                // Calculate Derivative based on Activation Type
                double dout_dZ = 1.0; // Default for LINEAR (f'(x) = 1)
                
                if (activationType == Activation::SIGMOID) {
                    dout_dZ = output * (1.0 - output);
                } else if (activationType == Activation::RELU) {
                    dout_dZ = output > 0 ? 1.0 : 0.0;
                }
                
                // Delta (dL/dZ)
                double delta = current_dL_dout * dout_dZ;
                
                // Calculate error to pass backward (MUST do this before updating weights!)
                // The blame for the previous layer's input is delta * weight
                for (size_t i = 0; i < inputSize; i++) {
                    size_t w_index = n * inputSize + i;
                    double current_w = this->weights.getIndexMem(w_index);
                    
                    double current_dL_dInput = dL_dInput.getIndexMem(i);
                    dL_dInput.setIndexMem(i, current_dL_dInput + (delta * current_w));
                }
                
                // Update Bias
                double current_b = this->bias.getIndexMem(n);
                this->bias.setIndexMem(n, current_b - (this->learningRate * delta));
                
                // Update Weights
                for (size_t i = 0; i < inputSize; i++) {
                    size_t w_index = n * inputSize + i;
                    double current_w = this->weights.getIndexMem(w_index);
                    double grad_w = delta * input.getIndexMem(i);
                    this->weights.setIndexMem(w_index, current_w - (this->learningRate * grad_w));
                }
            }
            
            // Return the error signal so the previous layer can use it!
            return dL_dInput;
        }

        void load(const std::string&filename){
            std::ifstream model_weights(filename);

            if(!model_weights){
                std::cerr << "Error : could not load the model weights" << std::endl;
                return;
            }

            std::string line;

            std::vector<size_t> shape;

            size_t lineCount = 1;
            while(std::getline(model_weights, line)){
                std::stringstream ss(line);

                if(lineCount == 1){
                    size_t number;
                    while(ss >> number){
                        shape.push_back(number);
                    }
                    this->weights.resize(shape);
                } else if(lineCount == 2){
                    double number;
                    size_t i = 0;
                    while(ss >> number){
                        this->weights.setIndexMem(i++, number);
                    }
                } else {
                    double number;
                    size_t i = 0; 
                    while(ss >> number){
                        this->bias.setIndexMem(i++, number);
                    }
                }
                lineCount++;
            }

            model_weights.close();
            std::cout << "Model weights has been loaded successfully" << std::endl;
        }

        void save(const std::string& filename){
            std::ofstream model_weights(filename);

            if(!model_weights){
                std::cerr << "Error : Could not save the model weigths to a file" << std::endl;
                return;
            }

            /*
            // format of the model file
            
                metadata - shape
                serialized weights (tensor)
                bias (tensor)
            */

            // first line : shape
            for(size_t i = 0; i < this->weights.getShape().size(); i++){
                model_weights << this->weights.getShape()[i] << " ";
            }
            model_weights << "\n";

            // serialized weights/parameters

            for(size_t i = 0; i < this->weights.getSize(); i++){
                model_weights << this->weights.getIndexMem(i) << " ";
            }
            model_weights << "\n";

            // serialized bias

            for(size_t i = 0; i < this->bias.getSize(); i++){
                model_weights << this->bias.getIndexMem(i) << " ";
            }
            model_weights << "\n";

            model_weights.close();

            std::cout << "Model saved successfully" << std::endl;
        }
};

int main(){
    std::string filename = "big_dataset.csv";
    std::ifstream file(filename);

    if(!file.is_open()){
        file.open("../" + filename);
    }

    if(!file.is_open()){
        std::cerr << "Failed to open " << filename << ". Please ensure it exists in your current directory.\n";
        return 1;
    }

    std::vector<std::pair<Tensor<double>, double>> dataset;
    std::string line;

    //skip head and count the columns
    std::getline(file, line);

    size_t numCols = 1;
    for(char c : line){
        if(c == ',') numCols++;
    }

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

    // --- DEEP LEARNING ---
    // Hidden Layer (w/ ReLU to learn non-linear patterns)
    Layer hidden_layer(8, inputFeatures);
    hidden_layer.setLearningRate(0.01); 
    hidden_layer.setActivation(Activation::RELU);

    Layer hidden_layer2(5, 8);
    hidden_layer2.setLearningRate(0.01); 
    hidden_layer2.setActivation(Activation::RELU);
    
    // Output Layer (w/ Sigmoid to output 0 or 1)
    Layer output_layer(1, 5);
    output_layer.setLearningRate(0.01);
    output_layer.setActivation(Activation::SIGMOID);
    
    size_t epochs = 40; // You can adjust this!
    
    std::cout << "Starting training on " << dataset.size() << " samples with " << inputFeatures << " features each...\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        
        for (const auto& data_point : dataset) {
            const Tensor<double>& x = data_point.first;
            double target_y = data_point.second;
            
            // 1. Forward Pass (Relay Race)
            Tensor<double> hidden_out = hidden_layer.forward(x);
            Tensor<double> hidden_out2 = hidden_layer2.forward(hidden_out);
            Tensor<double> prediction_tensor = output_layer.forward(hidden_out2);
            double prediction = prediction_tensor.getIndexMem(0);
            
            // 2. Calculate Error (MSE)
            double error = prediction - target_y;
            total_loss += 0.5 * error * error;
            
            // 3. Initial Gradient for Output Layer
            Tensor<double> dL_dout({1});
            dL_dout.setIndexMem(0, error);
            
            // 4. Backward Pass (Relay Race)

            Tensor<double> hidden_error2 = output_layer.backward(hidden_out2, dL_dout);
            Tensor<double> hidden_error = hidden_layer2.backward(hidden_out, hidden_error2);
            hidden_layer.backward(x, hidden_error);
        }
        
        // Print average loss every 10 epochs to watch it learn
        if (epoch % 10 == 0 || epoch == epochs - 1) {
            double avg_loss = total_loss / dataset.size();
            std::cout << "Epoch " << epoch << " | Average Loss: " << avg_loss << "\n";
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> training_time = end_time - start_time;
    
    std::cout << "Training complete in " << training_time.count() << " seconds!\n\n";
    
    // --- TESTING PHASE ---
    std::cout << "================ TESTING PHASE ================\n";
    double test_loss = 0.0;
    
    std::cout << std::left << std::setw(15) << "Prediction" 
              << std::left << std::setw(15) << "Target" 
              << "Error\n";
    std::cout << "-----------------------------------------------\n";
    
    for (size_t i = 0; i < dataset.size(); ++i) {
        const Tensor<double>& x = dataset[i].first;
        double target_y = dataset[i].second;
        
        // Forward pass
        Tensor<double> hidden_out = hidden_layer.forward(x);
        Tensor<double> hidden_out2 = hidden_layer2.forward(hidden_out);
        Tensor<double> prediction_tensor = output_layer.forward(hidden_out2);
        double prediction = prediction_tensor.getIndexMem(0);
        
        // Calculate MSE
        double error = prediction - target_y;
        test_loss += 0.5 * error * error;
        
        // Print first 10 samples
        if (i < 10) {
            std::cout << std::left << std::setw(15) << prediction 
                      << std::left << std::setw(15) << target_y 
                      << error << "\n";
        }
    }
    
    double avg_test_loss = test_loss / dataset.size();
    std::cout << "\nFinal Test MSE: " << avg_test_loss << "\n";
    std::cout << "===============================================\n";
    
    return 0;
}