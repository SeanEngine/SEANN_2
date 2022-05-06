//
// Created by Dylan on 5/1/2022.
//

#ifndef SEANN_2_PARAMETER_CUH
#define SEANN_2_PARAMETER_CUH

#include "../../seblas/tensor/Tensor.cuh"

using namespace seblas;

namespace seann {
    class Parameter {
    public:
        Tensor* a;
        Tensor* grad;

        static Parameter* create(shape4 dims){
            Parameter* p;
            cudaMallocHost(&p, sizeof(Parameter));
            p->a = Tensor::declare(dims)->create();
            p->grad = Tensor::declare(dims)->create();
            return p;
        };

        template<typename... Args>
        static Parameter* create(Args &&... args) {
            return create(shape4(std::forward<Args>(args)...));
        }

        static Parameter* declare(shape4 dims){
            Parameter* p;
            cudaMallocHost(&p, sizeof(Parameter));
            p->a = Tensor::declare(dims);
            p->grad = Tensor::declare(dims);
            return p;
        };

        template<typename... Args>
        static Parameter* declare(Args &&... args) {
            return declare(shape4(std::forward<Args>(args)...));
        }

        static Parameter* create(Tensor* src){
            Parameter* p;
            cudaMallocHost(&p, sizeof(Parameter));
            p->a = src;
            p->grad = Tensor::declare(src->dims)->create();
            return p;
        };

        //inherit keeps the shape of the parameter
        //but uses the same element memory block as another parameter
        Parameter* inherit(Parameter* src){
            assert(src->a->dims.size == this->a->dims.size);
            this->a->attach(src->a);
            this->grad->attach(src->grad);
            return this;
        }
    };
} // seann

#endif //SEANN_2_PARAMETER_CUH
