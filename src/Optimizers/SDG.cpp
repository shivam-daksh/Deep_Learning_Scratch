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
    }
}
