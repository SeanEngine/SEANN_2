//
// Created by Dylan on 5/11/2022.
//

#include "Softmax.cuh"

void seann::Softmax::forward() {
    softmax(X->a, Y->a, reduceBuffer, INPUT_SIZE);
}

void seann::Softmax::xGrads() {
    //Y->grad = Y->a - correct
    //this is controlled by loss function
    Y->grad->copyD2D(X->grad);
}
