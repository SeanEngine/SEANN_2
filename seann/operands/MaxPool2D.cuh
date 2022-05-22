//
// Created by Dylan on 5/21/2022.
//

#ifndef SEANN_2_MAXPOOL2D_CUH
#define SEANN_2_MAXPOOL2D_CUH

#include "OperandBase.cuh"

namespace seann {
    class MaxPool2D : public OperandBase{
    public:
        Tensor* record;
        explicit MaxPool2D(shape4 input, uint32 stepH, uint32 stepW){
            X = Parameter::declare(input);
            Y = Parameter::create(input.n, input.c, input.h / stepH, input.w / stepW);
            record = Tensor::declare(input)->create();
        }

        void randFillNetParams() override{}

        string info() override;

        void paramGrads() override{}

        void xGrads() override;

        void forward() override;

        void updateParams() override{}

        void batchUpdateParams() override{}

        void initNetParams(OptimizerInfo *info) override{}
    };

} // seann

#endif //SEANN_2_MAXPOOL2D_CUH
