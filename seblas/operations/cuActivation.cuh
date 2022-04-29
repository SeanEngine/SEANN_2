//
// Created by Dylan on 4/28/2022.
//

#ifndef SEANN_2_CUACTIVATION_CUH
#define SEANN_2_CUACTIVATION_CUH

#include "../tensor/Tensor.cuh"
/**
 * Activation functions and their derivatives
 */
namespace seblas {

    // relu activation
    Tensor* relu(Tensor* X, Tensor* outX);

    // relu derivative
    Tensor* reluGrad(Tensor* X, Tensor* outX);

    // lRelu activation
    Tensor* lRelu(Tensor* X, Tensor* outX, float alpha);

    // lRelu derivative
    Tensor* lReluGrad(Tensor* X, Tensor* outX, float alpha);

    // elu activation
    Tensor* elu(Tensor* X, Tensor* outX, float alpha);

    // elu derivative
    Tensor* eluGrad(Tensor* X, Tensor* outX, float alpha);

    // sigmoid activation
    Tensor* sigmoid(Tensor* X, Tensor* outX);

    // sigmoid derivative
    Tensor* sigmoidGrad(Tensor* X, Tensor* outX);

    // tanh activation
    Tensor* tanh(Tensor* X, Tensor* outX);

    // tanh derivative
    Tensor* tanhGrad(Tensor* X, Tensor* outX);
}

#endif //SEANN_2_CUACTIVATION_CUH
