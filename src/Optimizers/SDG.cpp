#include "SGD.hh"
#include "Parameter.hh"

SGD::SGD(Model &model, double lr) : model(model), learningRate(lr){}

void SGD::step(){
    std::vector<Layer>& layers = this->model.getLayers();

    for(Layer &layer : layers){
       std::vector<Parameter<double>*> parameters = layer.getParameter();

       for(Parameter<double>* p : parameters){
        double* data = p->value.data();
        const double* grad = p->gradient.data();
        size_t sz = p->value.getSize();

        for(size_t i = 0; i < sz; ++i){
            data[i] -= (this->learningRate * grad[i]);
        }
       }
        // Tensor<double> &weights = wt.value;
        // Tensor<double> &dW = wt.gradient;
        
        // Parameter<double> &b = layer.getParameter(ParamType::BIAS);
        // Tensor<double> &bias = b.value;
        // Tensor<double> &dB = b.gradient;

        // size_t no_neurons = weights.getShape()[0];
        // size_t inputSize = weights.getShape()[1];

        // for(size_t neuron = 0; neuron < no_neurons; neuron++){

        //     //update weights
        //     for(size_t i = 0; i < inputSize; i++){
        //         size_t wt_ind = neuron * inputSize + i;
        //         double old_wt = weights.getIndexMem(wt_ind);
        //         weights.setIndexMem(wt_ind, old_wt - this->learningRate * dW.getIndexMem(wt_ind));
        //     }

        //     // update bias
        //     double old_bias = bias.getIndexMem(neuron);
        //     bias.setIndexMem(neuron, old_bias - this->learningRate * dB.getIndexMem(neuron));
        // }
    }
}
