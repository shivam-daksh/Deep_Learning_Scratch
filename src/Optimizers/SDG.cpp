#include "SGD.hh"

SGD::SGD(Model &model, double lr) : model(model), learningRate(lr){}

void SGD::step(){
    std::vector<Layer>& layers = this->model.getLayers();

    for(Layer &layer : layers){
        Tensor<double> &weights = layer.getWeights();
        Tensor<double> &bias =  layer.getBias();
        const Tensor<double> &dW = layer.getdW();
        const Tensor<double> &dB = layer.getdB();

        size_t no_neurons = weights.getShape()[0];
        size_t inputSize = weights.getShape()[1];

        for(size_t neuron = 0; neuron < no_neurons; neuron++){

            //update weights
            for(size_t i = 0; i < inputSize; i++){
                size_t wt_ind = neuron * inputSize + i;
                double old_wt = weights.getIndexMem(wt_ind);
                weights.setIndexMem(wt_ind, old_wt - this->learningRate * dW.getIndexMem(wt_ind));
            }

            // update bias
            double old_bias = bias.getIndexMem(neuron);
            bias.setIndexMem(neuron, old_bias - this->learningRate * dB.getIndexMem(neuron));
        }
    }
}
