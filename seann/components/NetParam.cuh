//
// Created by Dylan on 5/5/2022.
//

#ifndef SEANN_2_NETPARAM_CUH
#define SEANN_2_NETPARAM_CUH

#include "Parameter.cuh"
#include "../optimizers/Optimizer.cuh"


namespace seann {
    /**
     * Netparams are parameters that are actually upgrading
     * such as weights and biases.
     * each net param will contain a optimizer
     */
    class NetParam {
    public:
        Parameter* A;
        Optimizer* opt;

        NetParam(Parameter* A, Optimizer* opt) : A(A), opt(opt) {}

        NetParam(Parameter* A, OptimizerInfo* info) : A(A) {
            opt = info->create(A);
        }

        NetParam(shape4 shape, OptimizerInfo* info) {
            A = Parameter::create(shape);
            opt = info->create(A);
        }

        template<typename... Args>
        explicit NetParam(OptimizerInfo* info, Args&&... args) {
            A = Parameter::create(std::forward<Args>(args)...);
            opt = info->create(A);
        }
    };
}

#endif //SEANN_2_NETPARAM_CUH
