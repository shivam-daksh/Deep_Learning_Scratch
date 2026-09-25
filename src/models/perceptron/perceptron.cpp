#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include "Tensor.hh"

class Perceptron{
    private:
        Tensor<double> weights;
        double bias;
        double learningRate = 0.01;
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

            double res = dot_product.sum()({});
            res += bias;
            return res;
        }

        void train(const Tensor<double>& input, const double& target){
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

        void load(const std::string& filename){
            std::ifstream model(filename);

            if(!model){
                std::cerr << "Error : could not save to model weights" << std::endl;
                return;
            }

            std::string line;

            int lineCount = 1;
            std::vector<size_t> shape;
            // Tensor<double> weights;
            // double bias;

            while(std::getline(model, line)){
                
                std::stringstream ss(line);

                if(lineCount == 1){
                    size_t number;
                    while(ss >> number){
                        shape.push_back(number);
                    }
                    this->weights.resize(shape);
                } else if(lineCount == 2){
                    // weights
                    double number;
                    size_t i = 0;
                    while(ss >> number){
                        this->weights.setIndexMem(i++, number);
                    }
                } else {
                    // bias
                    ss >> this->bias;
                }
                lineCount++;
            }

            model.close();
            std::cout << "Model weights has been loaded" << std::endl;

        }

        void save(const std::string& filename){
            std::ofstream model(filename);

            if(!model){
                std::cerr << "Error : could not save to model weights" << std::endl;
                return;
            }

            // format of the model file
            /*
            metadata - shape
            serialized weights (tensor)
            bias
            */

            for(size_t i = 0; i < this->weights.getShape().size(); i++){
                model << weights.getShape()[i] << " ";
            }

            model << "\n";

            // now serialized contiguous weights

            for(size_t i = 0; i < this->weights.getSize(); i++){
                model << this->weights.getIndexMem(i) << " ";
            }
            model << "\n";

            // bias values

            model << this->bias << "\n";


            model.close();

            std::cout << "Model saved successfully" << std::endl;
        }


    };

int main() {
    std::ifstream file("dataset.csv");
    if (!file.is_open()) {
        file.open("../dataset.csv"); // Try looking in the parent directory if run from build/
    }
    
    if (!file.is_open()) {
        std::cerr << "Failed to open dataset.csv. Please ensure it exists in your current directory.\n";
        return 1;
    }

    std::vector<std::pair<Tensor<double>, double>> dataset;
    std::string line;
    
    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string val;
        
        std::getline(ss, val, ',');
        double x1 = std::stod(val);
        
        std::getline(ss, val, ',');
        double x2 = std::stod(val);
        
        std::getline(ss, val, ',');
        double y = std::stod(val);

        Tensor<double> input({2});
        input({0}) = x1;
        input({1}) = x2;
        
        dataset.push_back({input, y});
    }

    // Create a perceptron with 2 input features
    Perceptron p(2);
    p.setLearningRate(0.01);
    
    int epochs = 10;
    std::cout << "Training for " << epochs << " epochs...\n";

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double totalLoss = 0.0;
        for (const auto& data : dataset) {
            double predicted = p.forward(data.first);
            double error = predicted - data.second;
            totalLoss += error * error;
            p.train(data.first, data.second);
        }
        
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " Loss: " << (totalLoss / dataset.size()) << "\n";
        }
    }

    std::cout << "\nFinal Predictions vs Targets:\n";
    for (const auto& data : dataset) {
        double predicted = p.forward(data.first);
        std::cout << "Input: (" << data.first({0}) << ", " << data.first({1}) 
                  << ") | Predicted: " << predicted 
                  << " | Target: " << data.second << "\n";
    }

    return 0;
}