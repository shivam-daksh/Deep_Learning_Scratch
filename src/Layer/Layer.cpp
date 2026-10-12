#include "Layer.hh"
#include <cmath>
#include <iostream>

Layer::Layer(size_t no_of_neurons, size_t inputSize) : weights({no_of_neurons, inputSize}), bias({no_of_neurons}), dW({no_of_neurons, inputSize}), dB({no_of_neurons}), last_input({0}), last_output({0}) {
    // Random weight initialization (CRITICAL to break symmetry in Deep Networks!)
    for(size_t i = 0; i < weights.getSize(); i++) {
        double r = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // Random between -1 and 1
        weights.setIndexMem(i, r);
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
    for(size_t i = 0; i < res.getSize(); i++) {
        double val = res.getIndexMem(i);
        if (activationType == Activation::SIGMOID) {
            res.setIndexMem(i, 1.0 / (1.0 + std::exp(-val)));
        } else if (activationType == Activation::RELU) {
            res.setIndexMem(i, val > 0 ? val : 0.0);
        }
    }
    
    this->last_output = res;
    return res;
}

Tensor<double> Layer::backward(const Tensor<double>& dL_dout) {
    // 1. Calculate the local gradient of the activation function (dout_dZ)
    Tensor<double> dL_dZ(this->last_output.getShape());
    for (size_t i = 0; i < this->last_output.getSize(); i++) {
        double output = this->last_output.getIndexMem(i);
        double grad = dL_dout.getIndexMem(i);
        
        double dout_dZ = 1.0; // Default LINEAR
        if (activationType == Activation::SIGMOID) {
            dout_dZ = output * (1.0 - output);
        } else if (activationType == Activation::RELU) {
            dout_dZ = output > 0 ? 1.0 : 0.0;
        }
        
        // Element-wise multiplication: dL_dZ = dL_dout * dout_dZ
        dL_dZ.setIndexMem(i, grad * dout_dZ);
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
