//
// Created by Dylan on 5/1/2022.
//

#ifndef SEANN_2_OPERANDBASE_CUH
#define SEANN_2_OPERANDBASE_CUH

#include "../components/Parameter.cuh"

namespace seann {
    class OperandBase {
    public:
        Parameter* X;  //input
        Parameter* Y;  //output

        //calculate : X -> Y
        virtual void forward() = 0;

        virtual void paramGrads() = 0;

        virtual void xGrads() = 0;

        virtual void updateX() = 0;

        virtual void updateParams() = 0;
    };
}

#endif //SEANN_2_OPERANDBASE_CUH
