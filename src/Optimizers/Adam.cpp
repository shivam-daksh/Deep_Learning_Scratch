#include "Adam.hh"
#include "Parameter.hh"

Adam::Adam(Model &model, const double beta1, const double beta2, const double lr) : model(model), beta1(beta1), beta2(beta2), learningRate(lr){

}

void Adam::step(){
    std::vector<std::unique_ptr<Layer>>& layers = this->model.getLayers();
    size_t layer_count = 0;
    for(std::unique_ptr<Layer> &layer : layers){

        std::vector<Parameter<double>*> parameters = layer->getParameter();

        // Tensor<double> &w_vel_layer = this->vel_w[layer_count];
        // Tensor<double> &b_vel_layer = this->vel_b[layer_count];

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

            for(size_t i = 0; i < sz; ++i){
                // first moment
                double mt = (this->beta1 * moment[i]) + ((1 - this->beta1) * grad[i]);
                moment[i] = mt;
                // second moment
                double v = (this->beta2 * vel[i]) + ((1 - this->beta2) * (grad[i] * grad[i]));
                vel[i] = v;

                // bias correction
                mt /= (1 - this->beta1);
                v /= (1 - this->beta2);

                double val = (mt) / (std::sqrt(v) + 1.0e-8);
                
                data[i] -= (this->learningRate * val);
            }
        }


        // Parameter<double> &wt = layer.getParameter(ParamType::WEIGHT);
        // Tensor<double> &weights = wt.value;
        // Tensor<double> &dW = wt.gradient;
        
        // Parameter<double> &b = layer.getParameter(ParamType::BIAS);
        // Tensor<double> &bias = b.value;
        // Tensor<double> &dB = b.gradient;

        // Tensor<double> &w_vel_layer = this->vel_w[layer_count];
        // Tensor<double> &b_vel_layer = this->vel_b[layer_count];
        // Tensor<double> &w_m_layer = this->m_w[layer_count];
        // Tensor<double> &b_m_layer = this->m_b[layer_count];


        // const size_t no_neurons = weights.getShape()[0];
        // const size_t inputSize = weights.getShape()[1];

        // for(size_t n = 0; n < no_neurons; n++){

        //     for(size_t i = 0; i < inputSize; i++){
        //         size_t wt_ind = n * inputSize + i;
        //         double dw = dW.getIndexMem(wt_ind);
        //         // first moment
        //         double new_first_moment = (this->beta1 * w_m_layer.getIndexMem(wt_ind)) + ((1 - this->beta1) * dw);
        //         w_m_layer.setIndexMem(wt_ind, new_first_moment);
                
        //         // second moment
        //         double new_second_moment = (this->beta2 * w_vel_layer.getIndexMem(wt_ind)) + ((1 - this->beta2) * (dw * dw));
        //         w_vel_layer.setIndexMem(wt_ind, new_second_moment);

        //         // bias correction
        //         new_first_moment /= (1 - this->beta1);
        //         new_second_moment /= (1 - this->beta2);
        //         double bias_correction = (new_first_moment) / (std::sqrt(new_second_moment) + 1.0e-8);

        //         double old_wt = weights.getIndexMem(wt_ind);
        //         // update the weights
        //         weights.setIndexMem(wt_ind, old_wt - this->learningRate * bias_correction);
        //     }
            
        //     //get the gradient
        //     double db = dB.getIndexMem(n);

        //     // first moment
        //     double new_first_moment = (this->beta1 * b_m_layer.getIndexMem(n)) + ((1 - this->beta1) * db);
        //     b_m_layer.setIndexMem(n, new_first_moment);
            
        //     //second moment
        //     double new_second_moment = (this->beta2 * b_vel_layer.getIndexMem(n)) + ((1 - this->beta2) * (db * db));
        //     b_vel_layer.setIndexMem(n, new_second_moment);

        //     //bias correction
        //     new_first_moment /= (1 - this->beta1);
        //     new_second_moment /= (1 - this->beta2);
        //     double bias_correction = (new_first_moment) / (std::sqrt(new_second_moment) + 1.0e-8);
            
        //     double old_bias = bias.getIndexMem(n);
        //     bias.setIndexMem(n, old_bias - this->learningRate * bias_correction);
        // }

        layer_count++;
    }
}