#include <iostream>
#include <vector>
#include "Tensor.hh"

class Perceptron{
    private:
        Tensor<double> weights;
        double bias;
        double learningRate = 0.1;
        size_t epochs = 10;

    public:
        Perceptron(size_t inputSize) : weights({inputSize}), bias(1){}
        
        void setEpochs(size_t epochs){
            this->epochs = epochs;
        }

        void setLearningRate(double lr){
            this->learningRate = lr;
        }

        double forward(const Tensor<double>& input){
            Tensor<double> dot_product(input.getShape());
            dot_product = (weights * input);

            double res = dot_product.sum()({0});
            res += bias;
            return res;
        }

        Tensor<double> train(const Tensor<double>& input, const double& target){
            double predicted = this->forward(input);
            double error = predicted - target;
            // double L = error * error;

            double dL_dE = 2 * error;

            double dL_dz = dL_dE;

            // dL/dw
            // weight update
            for(size_t i = 0; i < weights.getSize(); i++){
                double dL_dw = (dL_dz) * (input({i}));
                weights({i}) -= (this->learningRate * dL_dw);
            }

            //bias update
            double dL_db = dL_dz;
            this->bias -= (this->learningRate * dL_db);            

        }
};

int main() {

    // One training example
    Tensor<double> input({2});
    input({0}) = 2.0;
    input({1}) = 3.0;

    // Desired output
    double target = 10.0;

    // Create a perceptron with 2 input features
    Perceptron p(2);

    // Check prediction before training
    double predictionBefore = p.forward(input);

    double errorBefore = predictionBefore - target;
    double lossBefore = errorBefore * errorBefore;

    std::cout << "Before training:\n";
    std::cout << "Prediction: " << predictionBefore << '\n';
    std::cout << "Loss:       " << lossBefore << '\n';

    // Perform ONE training step
    p.train(input, target);

    // Check prediction after training
    double predictionAfter = p.forward(input);

    double errorAfter = predictionAfter - target;
    double lossAfter = errorAfter * errorAfter;

    std::cout << "\nAfter one update:\n";
    std::cout << "Prediction: " << predictionAfter << '\n';
    std::cout << "Loss:       " << lossAfter << '\n';

    return 0;
}