//
// Created by Dylan on 5/11/2022.
//

#ifndef SEANN_2_SOFTMAX_CUH
#define SEANN_2_SOFTMAX_CUH

#include "OperandBase.cuh"

namespace seann {
    class Softmax : public OperandBase {
    public:
        uint32 INPUT_SIZE;
        Tensor* reduceBuffer;
        explicit Softmax(uint32 INPUT_SIZE) : INPUT_SIZE(INPUT_SIZE) {
            X = Parameter::declare(INPUT_SIZE, 1);
            Y = Parameter::create(INPUT_SIZE, 1);
            reduceBuffer = INPUT_SIZE / 1024 > 0 ? Tensor::declare(INPUT_SIZE,1)->create() : nullptr;
        }

        string info() override {
            return "Softmax { " + to_string(INPUT_SIZE) + " }";
        }

        void initNetParams(OptimizerInfo *info) override{}

        void forward() override;

        void xGrads() override;

        void batchUpdateParams() override{}

        void updateParams() override{}

        void paramGrads() override{}
    };
}


#endif //SEANN_2_SOFTMAX_CUH
