#include "Optimizer.hh"

class SGD : public Optimizer {
    private:
        double learningRate;
    
    public:
        explicit SGD(double lr);
        void step(Model &model) override;

        ~SGD() override = default;
};