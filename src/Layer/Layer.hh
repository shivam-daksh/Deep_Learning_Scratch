#pragma once
#include <cstdlib>
#include "Tensor.hh"
#include "Parameter.hh"

enum class Activation { SIGMOID, RELU, LINEAR };
enum class InitType {HE, XAVIER};
enum class ParamType {WEIGHT, BIAS};

class Layer {
    private:
        // double learningRate = 0.01;
        // parameters
        std::vector<Parameter<double>*> param_list;
        Parameter<double> weights;
        Parameter<double> bias;
        // Tensor<double> weights;
        // Tensor<double> bias;

        // gradient
        // Tensor<double> dW;
        // Tensor<double> dB;

        // Cache for backpropagation
        Tensor<double> last_input;
        Tensor<double> last_output;
        
        Activation activationType = Activation::LINEAR;

        inline void registerParam(Parameter<double>& p){
            this->param_list.push_back(&p);
        }

    public:
        // Layer(size_t no_of_neurons, size_t inputSize);
        Layer(const size_t no_of_neurons, const size_t inputSize, const InitType type);
        Layer(const Layer& other);
        
        // void setLearningRate(double lr);
        void setActivation(Activation type);
        void initWeights(const InitType type);
        
        Tensor<double> forward(const Tensor<double>& input);
        Tensor<double> backward(const Tensor<double>& dL_dout);
        std::vector<Parameter<double>*> getParameter();
        Tensor<double>& getWeights();
        Tensor<double>& getBias();
        Tensor<double>& getdW();
        Tensor<double>& getdB();
};
