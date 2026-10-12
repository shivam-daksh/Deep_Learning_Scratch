#pragma once
#include "Tensor.hh"
#include <cstdlib>

enum class Activation { SIGMOID, RELU, LINEAR };
enum class InitType {HE, XAVIER};

class Layer {
    private:
        // double learningRate = 0.01;
        // parameters
        Tensor<double> weights;
        Tensor<double> bias;

        // gradient
        Tensor<double> dW;
        Tensor<double> dB;

        // Cache for backpropagation
        Tensor<double> last_input;
        Tensor<double> last_output;
        
        Activation activationType = Activation::LINEAR;

    public:
        // Layer(size_t no_of_neurons, size_t inputSize);
        Layer(const size_t no_of_neurons, const size_t inputSize, const InitType type);
        
        // void setLearningRate(double lr);
        void setActivation(Activation type);
        void initWeights(const InitType type);
        
        Tensor<double> forward(const Tensor<double>& input);
        Tensor<double> backward(const Tensor<double>& dL_dout);
        Tensor<double>& getWeights();
        Tensor<double>& getBias();
        Tensor<double>& getdW();
        Tensor<double>& getdB();
};
