#include "Optimizer.hh"

class SGD : public Optimizer {
    private:
        double learningRate;
        Model &model;
    
    public:
        explicit SGD(Model &model, double lr);
        void step() override;

        ~SGD() override = default;
};