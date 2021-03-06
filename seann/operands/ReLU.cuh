//
// Created by Dylan on 5/6/2022.
//

#ifndef SEANN_2_RELU_CUH
#define SEANN_2_RELU_CUH

#include "OperandBase.cuh"

namespace seann {
    class ReLU : public OperandBase {
    public:
        uint32 INPUT_SIZE{};
        ReLU() {}

        string info() override {
            return "ReLU          { " + std::to_string(INPUT_SIZE) + " }";
        }

        void initNetParams(OptimizerInfo *info, shape4 inShape) override{
            INPUT_SIZE = inShape.size;
            X = Parameter::declare(inShape);
            Y = Parameter::create(inShape);
        }

        void forward() override;

        void xGrads() override;

        void batchUpdateParams() override{}

        void updateParams() override{}

        void paramGrads() override{}

        void randFillNetParams() override{}
    };
} // seann

#endif //SEANN_2_RELU_CUH
