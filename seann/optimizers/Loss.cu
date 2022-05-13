//
// Created by Dylan on 5/13/2022.
//
#include "Loss.cuh"

namespace seann{
    void CrossEntropyLoss(Parameter* Y, Tensor* label){
        Y->a->copyD2D(Y->grad);
        *Y->grad - label;
    }
}
