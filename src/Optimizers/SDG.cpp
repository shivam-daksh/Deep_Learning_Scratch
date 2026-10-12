#include "SGD.hh"
#include "Parameter.hh"

SGD::SGD(Model &model, double lr, double weight_decay, double l1) : model(model), learningRate(lr){
    this->weight_decay = weight_decay;
    this->l1_ratio = l1;
}

void SGD::step(){
    std::vector<std::unique_ptr<Layer>>& layers = this->model.getLayers();

    for(std::unique_ptr<Layer> &layer : layers){
       std::vector<Parameter<double>*> parameters = layer->getParameter();

       for(Parameter<double>* p : parameters){
        double* data = p->value.data();
        const double* grad = p->gradient.data();
        size_t sz = p->value.getSize();
        bool reg = p->regularize;

        for(size_t i = 0; i < sz; ++i){
            double g = this->applyRegularization(grad[i], data[i], reg);
            data[i] -= (this->learningRate * g);
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
