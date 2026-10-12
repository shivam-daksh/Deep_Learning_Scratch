#include <cmath>
#include <iostream>
#include <random>
#include "Layer.hh"

Layer::Layer(const size_t no_of_neurons, const size_t inputSize, const InitType type): weights({no_of_neurons, inputSize}), bias({no_of_neurons}), dW({no_of_neurons, inputSize}), dB({no_of_neurons}), last_input({0}), last_output({0}) {
    // weight initialization for layer
    this->initWeights(type);
}

void Layer::initWeights(const InitType type){
    const size_t no_of_neurons = this->weights.getShape()[0];
    const size_t inputSize = this->weights.getShape()[1];

    std::random_device rd;
    std::mt19937 gen(rd());

    if(type == InitType::XAVIER){
        double std_dev = std::sqrt(2.0 / (no_of_neurons + inputSize));
        std::normal_distribution<double> dist(0.0, std_dev);

        for(size_t i = 0; i < this->weights.getSize(); i++) {
            weights.setIndexMem(i, dist(gen));
        }

    } else if(type == InitType::HE){
        double std_dev = std::sqrt(2.0 / (inputSize));
        std::normal_distribution<double> dist(0.0, std_dev);

        for(size_t i = 0; i < this->weights.getSize(); i++) {
            weights.setIndexMem(i, dist(gen));
        }
    }
    for(size_t i = 0; i < bias.getSize(); i++) {
        bias.setIndexMem(i, 0.0);
    }
}
Tensor<double>& Layer::getWeights(){
    return this->weights;
}

Tensor<double>& Layer::getBias(){
    return this->bias;
}

Tensor<double>& Layer::getdW(){
    return this->dW;
}

Tensor<double>& Layer::getdB(){
    return this->dB;
}

// void Layer::setLearningRate(double lr) {
//     this->learningRate = lr;
// }

void Layer::setActivation(Activation type) {
    this->activationType = type;
}

Tensor<double> Layer::forward(const Tensor<double>& input) {
    this->last_input = input; // Cache input for backprop
    
    // Z = X * W^T + b
    // input is [B, I] (or [I]), weights is [N, I], weights.T() is [I, N]
    // res is [B, N]
    Tensor<double> res = input.matmul(this->weights.T());
    res = res + this->bias; // Broadcasting takes care of [B, N] + [N]
            
    // Activation Function
    double* r_data = res.data();
    size_t r_sz = res.getSize();
    if (activationType == Activation::SIGMOID) {
        for(size_t i = 0; i < r_sz; i++) {
            double val = r_data[i];
            if (val >= 0.0) {
                r_data[i] = 1.0 / (1.0 + std::exp(-val));
            } else {
                double exp_val = std::exp(val);
                r_data[i] = exp_val / (1.0 + exp_val);
            }
        }
    } else if (activationType == Activation::RELU) {
        for(size_t i = 0; i < r_sz; i++) {
            r_data[i] = r_data[i] > 0.0 ? r_data[i] : 0.0;
        }
    }
    
    this->last_output = res;
    return res;
}

Tensor<double> Layer::backward(const Tensor<double>& dL_dout) {
    // 1. Calculate the local gradient of the activation function (dout_dZ)
    Tensor<double> dL_dZ(this->last_output.getShape());
    const double* out_data = this->last_output.data();
    const double* dout_data = dL_dout.data();
    double* dz_data = dL_dZ.data();
    size_t sz = this->last_output.getSize();

    if (activationType == Activation::SIGMOID) {
        for (size_t i = 0; i < sz; i++) {
            double o = out_data[i];
            dz_data[i] = dout_data[i] * (o * (1.0 - o));
        }
    } else if (activationType == Activation::RELU) {
        for (size_t i = 0; i < sz; i++) {
            dz_data[i] = dout_data[i] * (out_data[i] > 0.0 ? 1.0 : 0.0);
        }
    } else {
        for (size_t i = 0; i < sz; i++) {
            dz_data[i] = dout_data[i];
        }
    }
    
    // 2. Gradients w.r.t Weights and Biases (Matrix Math!)
    // dW = dL_dZ^T * X  => [N, B] * [B, I] => [N, I]
    // If input was 1D [I], last_input is [I]. 
    // dL_dZ is [N]. dL_dZ.T() is [N, 1], last_input is [1, I].
    // To handle 1D and 2D seamlessly in matmul, we ensure they are 2D logic.
    bool is_1d = (this->last_input.getShape().size() == 1);
    
    if (is_1d) {
        // Fallback for current single-sample training (1D vectors)
        for (size_t n = 0; n < this->bias.getSize(); n++) {
            double delta = dL_dZ.getIndexMem(n);
            this->dB.setIndexMem(n, delta);
            for (size_t i = 0; i < this->last_input.getSize(); i++) {
                size_t w_index = n * this->last_input.getSize() + i;
                this->dW.setIndexMem(w_index, delta * this->last_input.getIndexMem(i));
            }
        }
        return dL_dZ.matmul(this->weights); // [N] * [N, I] -> [I]
    } else {
        // True Mini-Batch Matrix Math (2D)
        this->dW = dL_dZ.T().matmul(this->last_input);
        
        // dB = sum(dL_dZ, axis=0) => sum over batches
        this->dB = dL_dZ.sum(0);
        
        // 3. Gradients w.r.t Input (to pass to the previous layer)
        // dL_dX = dL_dZ * W => [B, N] * [N, I] => [B, I]
        return dL_dZ.matmul(this->weights);
    }
}
