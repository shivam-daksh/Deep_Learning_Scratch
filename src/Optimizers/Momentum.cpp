#include <vector>
#include "Momentum.hh"
#include "Layer.hh"

Momentum::Momentum(Model &model, const double beta, const double lr) : model(model), beta(beta), learningRate(lr){
    std::vector<Layer> &layers = this->model.getLayers();

    for(Layer &layer : layers){
        Tensor<double> &weights = layer.getWeights();

        const size_t no_neurons = weights.getShape()[0];
        const size_t inputSize = weights.getShape()[1];

        Tensor<double> temp_w({no_neurons, inputSize});
        this->vel_w.push_back(temp_w);

        Tensor<double> temp_b({no_neurons});
        this->vel_b.push_back(temp_b);
    }
}

void Momentum::step(){
    std::vector<Layer> &layers = this->model.getLayers();
    size_t layer_count = 0;
    for(Layer &layer : layers){
        Tensor<double> &weights = layer.getWeights();
        Tensor<double> &dW = layer.getdW();
        
        Tensor<double> &bias = layer.getBias();
        Tensor<double> &dB = layer.getdB();

        Tensor<double> &w_vel_layer = this->vel_w[layer_count];
        Tensor<double> &b_vel_layer = this->vel_b[layer_count];


        const size_t no_neurons = weights.getShape()[0];
        const size_t inputSize = weights.getShape()[1];

        for(size_t n = 0; n < no_neurons; n++){

            for(size_t i = 0; i < inputSize; i++){
                size_t wt_ind = n * inputSize + i;

                double dw = dW.getIndexMem(wt_ind);
                double newVelW = this->beta * w_vel_layer.getIndexMem(wt_ind) + dw;
                w_vel_layer.setIndexMem(wt_ind, newVelW);

                double old_wt = weights.getIndexMem(wt_ind);

                weights.setIndexMem(wt_ind, old_wt - this->learningRate * newVelW);
            }
            
            double db = dB.getIndexMem(n);
            double newVelB = this->beta * b_vel_layer.getIndexMem(n) + db;
            double old_bias = bias.getIndexMem(n);
            bias.setIndexMem(n, old_bias - this->learningRate * newVelB);
        }

        layer_count++;
    }
}