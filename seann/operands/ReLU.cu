//
// Created by Dylan on 5/6/2022.
//

#include "ReLU.cuh"

namespace seann {
    void ReLU::forward() {
        relu(X->a, Y->a);
    }

    // ∂C/∂Z = ∂C/∂a * ∂a/∂Z = ∂C/∂a * σ'(Z)
    void ReLU::xGrads() {
        *reluGrad(X->a, X->grad) * Y->grad;
    }
} // seann