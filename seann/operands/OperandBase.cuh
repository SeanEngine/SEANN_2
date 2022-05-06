//
// Created by Dylan on 5/1/2022.
//

#ifndef SEANN_2_OPERANDBASE_CUH
#define SEANN_2_OPERANDBASE_CUH

#include "../components/Parameter.cuh"
#include "../../seblas/operations/cuOperations.cuh"
#include "../components/NetParam.cuh"

using namespace seblas;

namespace seann {
    class OperandBase {
    public:
        Parameter* X{};  //input, a shadow of the output of prev operand
        Parameter* Y{};  //output

        OperandBase* prev;
        OperandBase* next;

        //calculate : X -> Y
        virtual void forward() = 0;

        //calculate grads for operand parameters : weights, bias, etc
        virtual void paramGrads() = 0;

        //calculate grads for operand input : X
        virtual void xGrads() = 0;

        //do the gradient decent with optimizers
        virtual void updateParams() = 0;
        virtual void batchUpdateParams() = 0;

        virtual void initNetParams(OptimizerInfo* info) = 0;

        //X should be bind to the Y of the previous operand
        void bindPrev(OperandBase* prevPtr) {
            this->prev = prevPtr;
            X->inherit(prevPtr->Y);
        }

        void bindNext(OperandBase* nextPtr) {
            this->next = nextPtr;
        }
    };
}

#endif //SEANN_2_OPERANDBASE_CUH
