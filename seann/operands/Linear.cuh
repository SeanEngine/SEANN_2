//
// Created by Dylan on 5/1/2022.
//

#ifndef SEANN_2_LINEAR_CUH
#define SEANN_2_LINEAR_CUH

#include "OperandBase.cuh"

namespace seann {
    class Linear : public OperandBase {
    public:
        NetParam* weights{};
        NetParam* biases{};
        uint32 INPUT_SIZE;
        uint32 OUTPUT_SIZE;

        Linear(uint32 INPUT_SIZE, uint32 OUTPUT_SIZE){
            this->INPUT_SIZE = INPUT_SIZE;
            this->OUTPUT_SIZE = OUTPUT_SIZE;
            X = Parameter::declare(INPUT_SIZE, 1);
            Y = Parameter::create(OUTPUT_SIZE, 1);
        }

        string info() override {
            return "Linear { " + to_string(INPUT_SIZE) + ", " + to_string(OUTPUT_SIZE) + " }";
        }

        void initNetParams(OptimizerInfo *info) override;

        void forward() override;

        void paramGrads() override;

        void updateParams() override;

        void batchUpdateParams() override;

        void xGrads() override;

        void randFillNetParams() override;
    };
} // seann

#endif //SEANN_2_LINEAR_CUH
