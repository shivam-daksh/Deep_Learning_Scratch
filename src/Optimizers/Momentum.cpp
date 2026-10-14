#include <vector>
#include "Momentum.hh"
#include "Layer.hh"
#include "Parameter.hh"

Momentum::Momentum(Model &model, const double beta, const double lr, double weight_decay, double l1) 
    : model(model), beta(beta), learningRate(lr) {
    this->weight_decay = weight_decay;
    this->l1_ratio = l1;
}

void Momentum::step(){
    std::vector<std::unique_ptr<Layer>>& layers = this->model.getLayers();
    size_t layer_count = 0;
    for(std::unique_ptr<Layer> &layer : layers){

        std::vector<Parameter<double>*> parameters = layer->getParameter();

        for(Parameter<double>* p : parameters){

            if(this->velocities.find(p) == this->velocities.end()){
                Tensor<double> vel(p->value.getShape());
                vel.fill(0.0);
                velocities[p] = vel;
            }

            double* data = p->value.data();
            const double* grad = p->gradient.data();
            size_t sz = p->value.getSize();
            double* vel = velocities[p].data();
            bool reg = p->regularize;

            for(size_t i = 0; i < sz; ++i){
                double g = this->applyRegularization(grad[i], data[i], reg);
                double newVel = this->beta * vel[i] + g;
                vel[i] = newVel;
                data[i] -= (this->learningRate * newVel);
            }
        }

        layer_count++;
    }
}