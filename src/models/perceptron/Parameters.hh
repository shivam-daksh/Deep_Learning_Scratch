#include "Tensor.hh"

class Parameter {
    private:
        Tensor<double> weights;
        Tensor<double> values;
    public:
        inline Tensor<double>& weights(){
            return this->weights;
        }
        inline Tensor<double>& values(){
            return this->values;
        }
};