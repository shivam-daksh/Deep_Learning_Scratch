#include <vector>
#include "Momentum.hh"
#include "Layer.hh"
#include "Parameter.hh"

Momentum::Momentum(Model &model, const double beta, const double lr) : model(model), beta(beta), learningRate(lr){

}

void Momentum::step(){
    std::vector<Layer> &layers = this->model.getLayers();
    size_t layer_count = 0;
    for(Layer &layer : layers){

        std::vector<Parameter<double>*> parameters = layer.getParameter();

        // Tensor<double> &w_vel_layer = this->vel_w[layer_count];
        // Tensor<double> &b_vel_layer = this->vel_b[layer_count];

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

            for(size_t i = 0; i < sz; ++i){
                double newVel = this->beta * vel[i] + grad[i];
                vel[i] = newVel;
                data[i] -= (this->learningRate * newVel);
            }
        }

        // Parameter<double> &wt = layer.getParameter(ParamType::WEIGHT);
        // Tensor<double> &weights = wt.value;
        // Tensor<double> &dW = wt.gradient;
        
        // Parameter<double> &b = layer.getParameter(ParamType::BIAS);
        // Tensor<double> &bias = b.value;
        // Tensor<double> &dB = b.gradient;


        // const size_t no_neurons = weights.getShape()[0];
        // const size_t inputSize = weights.getShape()[1];

        // for(size_t n = 0; n < no_neurons; n++){

        //     for(size_t i = 0; i < inputSize; i++){
        //         size_t wt_ind = n * inputSize + i;

        //         double dw = dW.getIndexMem(wt_ind);
        //         double newVelW = this->beta * w_vel_layer.getIndexMem(wt_ind) + dw;
        //         w_vel_layer.setIndexMem(wt_ind, newVelW);

        //         double old_wt = weights.getIndexMem(wt_ind);

        //         weights.setIndexMem(wt_ind, old_wt - this->learningRate * newVelW);
        //     }
            
        //     double db = dB.getIndexMem(n);
        //     double newVelB = this->beta * b_vel_layer.getIndexMem(n) + db;
        //     b_vel_layer.setIndexMem(n, newVelB);
        //     double old_bias = bias.getIndexMem(n);
        //     bias.setIndexMem(n, old_bias - this->learningRate * newVelB);
        // }

        layer_count++;
    }
}