#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include "Tensor.hh"


class Layer {
    private:
        Tensor<double> weights;
        Tensor<double> bias;
        double learningRate = 0.01;
        size_t epochs = 10;
        
        // Cache the output for the backward pass
        Tensor<double> last_output;

    public:
        Layer(size_t no_of_neurons, size_t inputSize) : weights({no_of_neurons, inputSize}), bias({no_of_neurons}){}
        
        void setEpochs(size_t epochs){
            this->epochs = epochs;
        }

        void setLearningRate(double lr){
            this->learningRate = lr;
        }

        Tensor<double> forward(const Tensor<double>&input){
            Tensor<double> res = this->weights.matmul(input);
            res = res + this->bias;
            
            // Apply Sigmoid activation function
            for(size_t i = 0; i < res.getSize(); i++){
                double val = res.getIndexMem(i);
                res.setIndexMem(i, 1.0 / (1.0 + std::exp(-val)));
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
                
                // Activation (Sigmoid: dout/dZ = output * (1 - output))
                double dout_dZ = output * (1.0 - output);
                
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
    std::string filename = "dataset4d.csv";
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

    // Create a neuron layer (1 output neuron, inputFeatures inputs)
    Layer layer1(1, inputFeatures);
    layer1.setLearningRate(0.01);
    
    size_t epochs = 100; // You can adjust this!
    
    std::cout << "Starting training on " << dataset.size() << " samples with " << inputFeatures << " features each...\n";
    
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        
        for (const auto& data_point : dataset) {
            const Tensor<double>& x = data_point.first;
            double target_y = data_point.second;
            
            // 1. Forward pass
            Tensor<double> prediction_tensor = layer1.forward(x);
            double prediction = prediction_tensor.getIndexMem(0); // single neuron output
            
            // 2. Calculate error and loss (Mean Squared Error)
            double error = prediction - target_y;
            total_loss += 0.5 * error * error;
            
            // 3. Setup initial gradient for the backward pass (dL_dout)
            Tensor<double> dL_dout({1});
            dL_dout.setIndexMem(0, error);
            
            // 4. Backward pass
            layer1.backward(x, dL_dout);
        }
        
        // Print average loss every 10 epochs to watch it learn
        if (epoch % 10 == 0 || epoch == epochs - 1) {
            double avg_loss = total_loss / dataset.size();
            std::cout << "Epoch " << epoch << " | Average Loss: " << avg_loss << "\n";
        }
    }
    
    std::cout << "Training complete!\n";
    return 0;
}