#include <iostream>
#include <vector>
#include "Tensor.hh"

class Perceptron{
    private:
        Tensor<double> weights;
        double bias;

    public:
        Perceptron(size_t inputSize) : weights({inputSize}), bias(0.0){}

        double forward(const Tensor<double>& input){

        }
};
