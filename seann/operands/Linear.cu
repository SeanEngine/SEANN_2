//
// Created by Dylan on 5/1/2022.
//

#include "Linear.cuh"

namespace seann {
    void Linear::initNetParams(OptimizerInfo *info) {
        weights = new NetParam(info, OUTPUT_SIZE, INPUT_SIZE);
        biases = new NetParam(info, OUTPUT_SIZE, 1);
    }

    // a[l] = w[l] * a[l-1] + b[l]
    void Linear::forward() {
        sgemm(weights->A->a, X->a, Y->a);
        Y->a = *Y->a + biases->A->a;
    }

    void Linear::paramGrads() {
        // ∂w = error * a^T
        sgemmNTA(Y->grad, X->a, weights->A->grad);
        // ∂b = error
        Y->grad->copyD2D(biases->A->grad);
    }

    void Linear::updateParams() {
        weights->opt->apply();
        biases->opt->apply();
    }

    void Linear::batchUpdateParams() {
        weights->opt->batchApply();
        biases->opt->batchApply();
    }

    void Linear::xGrads() {
        // ∂x = w^T * ∂z
        sgemmTN(weights->A->a, Y->grad, X->grad);
    }
} // seann