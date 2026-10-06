#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "Tensor.hh"


class Model{
    private:
        double learningRate;
        size_t epochs;
    
    public:
        Model(const double lr, size_t epochs) : learningRate(lr), epochs(epochs){

        }

        void train(){
            
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

    // Create a neuron layer
}