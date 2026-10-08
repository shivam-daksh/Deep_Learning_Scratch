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

void Layer::setLearningRate(double lr) {
    this->learningRate = lr;
}

void Layer::setActivation(Activation type) {
    this->activationType = type;
}

Tensor<double> Layer::forward(const Tensor<double>& input) {
    this->last_input = input; // Cache input for backprop
    
    Tensor<double> res = this->weights.matmul(input);
    res = res + this->bias;
    
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
    size_t inputSize = this->last_input.getSize();
    size_t numNeurons = this->bias.getSize();
    
    Tensor<double> dL_dInput(this->last_input.getShape());
    for(size_t i = 0; i < inputSize; i++) {
        dL_dInput.setIndexMem(i, 0.0);
    }

    for (size_t n = 0; n < numNeurons; n++) {
        double output = this->last_output.getIndexMem(n);
        double current_dL_dout = dL_dout.getIndexMem(n);
        
        // Calculate Derivative based on Activation Type
        double dout_dZ = 1.0; // Default for LINEAR
        
        if (activationType == Activation::SIGMOID) {
            dout_dZ = output * (1.0 - output);
        } else if (activationType == Activation::RELU) {
            dout_dZ = output > 0 ? 1.0 : 0.0;
        }
        
        double delta = current_dL_dout * dout_dZ;
        
        for (size_t i = 0; i < inputSize; i++) {
            size_t w_index = n * inputSize + i;
            double current_w = this->weights.getIndexMem(w_index);
            
            // Pass error backward
            double current_dL_dInput = dL_dInput.getIndexMem(i);
            dL_dInput.setIndexMem(i, current_dL_dInput + (delta * current_w));
            
            // Update weight
            double input_val = this->last_input.getIndexMem(i);
            // this->weights.setIndexMem(w_index, current_w - (this->learningRate * delta * input_val));
            this->dW.setIndexMem(w_index, delta * input_val);
        }
        
        // Update Bias
        double current_b = this->bias.getIndexMem(n);
        // this->bias.setIndexMem(n, current_b - (this->learningRate * delta));
        this->dB.setIndexMem(n, delta);
    }
    
    return dL_dInput;
}
