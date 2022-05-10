//
// Created by Dylan on 5/6/2022.
//

#ifndef SEANN_2_RELU_CUH
#define SEANN_2_RELU_CUH

#include "OperandBase.cuh"

namespace seann {
    class ReLU : public OperandBase {
    public:
        uint32 INPUT_SIZE;
        explicit ReLU(uint32 INPUT_SIZE) : INPUT_SIZE(INPUT_SIZE) {
            X = Parameter::declare(INPUT_SIZE, 1);
            Y = Parameter::create(INPUT_SIZE, 1);
        }

        void initNetParams(OptimizerInfo *info) override{}

        void forward() override;

        void xGrads() override;

        void batchUpdateParams() override{}

        void updateParams() override{}

        void paramGrads() override{}
    };
} // seann

#endif //SEANN_2_RELU_CUH
