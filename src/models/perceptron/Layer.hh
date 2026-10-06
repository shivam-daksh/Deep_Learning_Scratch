#pragma once
#include "Tensor.hh"
#include <cstdlib>

enum class Activation { SIGMOID, RELU, LINEAR };

class Layer {
    private:
        Tensor<double> weights;
        Tensor<double> bias;
        Activation activationType = Activation::LINEAR;
        double learningRate = 0.01;
        
        // Cache for backpropagation
        Tensor<double> last_input;
        Tensor<double> last_output;

    public:
        Layer(size_t no_of_neurons, size_t inputSize);
        
        void setLearningRate(double lr);
        void setActivation(Activation type);
        
        Tensor<double> forward(const Tensor<double>& input);
        Tensor<double> backward(const Tensor<double>& dL_dout);
};
