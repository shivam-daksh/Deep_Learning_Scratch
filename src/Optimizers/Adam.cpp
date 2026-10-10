#include "Adam.hh"

Adam::Adam(Model &model, const double beta1, const double beta2, const double lr) : model(model), beta1(beta1), beta2(beta2), learningRate(lr){
    std::vector<Layer> &layers = this->model.getLayers();

    for(Layer &layer : layers){
        Tensor<double> &weights = layer.getWeights();

        const size_t no_neurons = weights.getShape()[0];
        const size_t inputSize = weights.getShape()[1];

        Tensor<double> temp_w({no_neurons, inputSize});
        this->vel_w.push_back(temp_w);
        this->m_w.push_back(temp_w);

        Tensor<double> temp_b({no_neurons});
        this->vel_b.push_back(temp_b);
        this->m_b.push_back(temp_b);
    }
}

void Adam::step(){
    std::vector<Layer> &layers = this->model.getLayers();
    size_t layer_count = 0;
    for(Layer &layer : layers){
        Tensor<double> &weights = layer.getWeights();
        Tensor<double> &dW = layer.getdW();
        
        Tensor<double> &bias = layer.getBias();
        Tensor<double> &dB = layer.getdB();

        Tensor<double> &w_vel_layer = this->vel_w[layer_count];
        Tensor<double> &b_vel_layer = this->vel_b[layer_count];
        Tensor<double> &w_m_layer = this->m_w[layer_count];
        Tensor<double> &b_m_layer = this->m_b[layer_count];


        const size_t no_neurons = weights.getShape()[0];
        const size_t inputSize = weights.getShape()[1];

        for(size_t n = 0; n < no_neurons; n++){

            for(size_t i = 0; i < inputSize; i++){
                size_t wt_ind = n * inputSize + i;
                double dw = dW.getIndexMem(wt_ind);
                // first moment
                double new_first_moment = (this->beta1 * w_m_layer.getIndexMem(wt_ind)) + ((1 - this->beta1) * dw);
                w_m_layer.setIndexMem(wt_ind, new_first_moment);
                
                // second moment
                double new_second_moment = (this->beta2 * w_vel_layer.getIndexMem(wt_ind)) + ((1 - this->beta2) * (dw * dw));
                w_vel_layer.setIndexMem(wt_ind, new_second_moment);

                // bias correction
                new_first_moment /= (1 - this->beta1);
                new_second_moment /= (1 - this->beta2);
                double bias_correction = (new_first_moment) / (std::sqrt(new_second_moment) + 1.0e-8);

                double old_wt = weights.getIndexMem(wt_ind);
                // update the weights
                weights.setIndexMem(wt_ind, old_wt - this->learningRate * bias_correction);
            }
            
            //get the gradient
            double db = dB.getIndexMem(n);

            // first moment
            double new_first_moment = (this->beta1 * b_m_layer.getIndexMem(n)) + ((1 - this->beta1) * db);
            b_m_layer.setIndexMem(n, new_first_moment);
            
            //second moment
            double new_second_moment = (this->beta2 * b_vel_layer.getIndexMem(n)) + ((1 - this->beta2) * (db * db));
            b_vel_layer.setIndexMem(n, new_second_moment);

            //bias correction
            new_first_moment /= (1 - this->beta1);
            new_second_moment /= (1 - this->beta2);
            double bias_correction = (new_first_moment) / (std::sqrt(new_second_moment) + 1.0e-8);
            
            double old_bias = bias.getIndexMem(n);
            bias.setIndexMem(n, old_bias - this->learningRate * bias_correction);
        }

        layer_count++;
    }
}