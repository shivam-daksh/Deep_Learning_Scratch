#include "Adam.hh"
#include "Parameter.hh"

Adam::Adam(Model &model, const double beta1, const double beta2, const double lr, double weight_decay, double l1) 
    : model(model), beta1(beta1), beta2(beta2), learningRate(lr) {
    this->weight_decay = weight_decay;
    this->l1_ratio = l1;
}

void Adam::step(){
    std::vector<std::unique_ptr<Layer>>& layers = this->model.getLayers();
    size_t layer_count = 0;
    for(std::unique_ptr<Layer> &layer : layers){

        std::vector<Parameter<double>*> parameters = layer->getParameter();

        for(Parameter<double>* p : parameters){

            if(this->velocities.find(p) == this->velocities.end()){
                Tensor<double> moment(p->value.getShape());
                Tensor<double> vel(p->value.getShape());
                moment.fill(0.0);
                vel.fill(0.0);
                velocities[p] = {moment, vel};
            }

            double* data = p->value.data();
            const double* grad = p->gradient.data();
            size_t sz = p->value.getSize();
            double* moment = velocities[p].first.data();
            double* vel = velocities[p].second.data();
            bool reg = p->regularize;

            for(size_t i = 0; i < sz; ++i){
                double g = this->applyRegularization(grad[i], data[i], reg);
                // first moment
                double mt = (this->beta1 * moment[i]) + ((1.0 - this->beta1) * g);
                moment[i] = mt;
                // second moment
                double v = (this->beta2 * vel[i]) + ((1.0 - this->beta2) * (g * g));
                vel[i] = v;

                // bias correction
                double mt_corr = mt / (1.0 - this->beta1);
                double v_corr = v / (1.0 - this->beta2);

                double val = (mt_corr) / (std::sqrt(v_corr) + 1.0e-8);
                
                data[i] -= (this->learningRate * val);
            }
        }

        layer_count++;
    }
}