#include "Tensor.hh"
#include "Model.hh"

// Interface for optimizer
class Optimizer {
    public:
        virtual void step(Model &model) = 0;
        virtual ~Optimizer() = default;
};