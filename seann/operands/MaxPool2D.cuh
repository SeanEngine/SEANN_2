//
// Created by Dylan on 5/21/2022.
//

#ifndef SEANN_2_MAXPOOL2D_CUH
#define SEANN_2_MAXPOOL2D_CUH

#include "OperandBase.cuh"

namespace seann {
    class MaxPool2D : public OperandBase{
    public:
        Tensor* record{};
        uint32 stepH;
        uint32 stepW;
        explicit MaxPool2D(uint32 stepH, uint32 stepW){
            this->stepH = stepH;
            this->stepW = stepW;
        }

        void randFillNetParams() override{}

        string info() override;

        void paramGrads() override{}

        void xGrads() override;

        void forward() override;

        void updateParams() override{}

        void batchUpdateParams() override{}

        void initNetParams(OptimizerInfo *info, shape4 inShape) override{
            X = Parameter::declare(inShape);
            Y = Parameter::create(inShape.n, inShape.c, inShape.h / stepH, inShape.w / stepW);
            record = Tensor::declare(inShape)->create();
        }
    };

} // seann

#endif //SEANN_2_MAXPOOL2D_CUH
