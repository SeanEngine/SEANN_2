//
// Created by Dylan on 5/25/2022.
//

#ifndef SEANN_2_CUPARALLEL_CUH
#define SEANN_2_CUPARALLEL_CUH


#include "../tensor/Tensor.cuh"

namespace seblas {
    //A special pack of functions to process
    //parallel input data (The n dimension)
    Tensor* paraAdd(Tensor* A, Tensor* B, Tensor* C);
    Tensor* paraAdd(Tensor* A, Tensor* B);

    Tensor* paraSub(Tensor* A, Tensor* B, Tensor* C);
    Tensor* paraSub(Tensor* A, Tensor* B);

    //add all elements in the nth dimension onto the resultant matrix
    Tensor* paraCumulate(Tensor* A, Tensor* B, Tensor* C);
    Tensor* paraCumulate(Tensor* A, Tensor* B);

    Tensor* batchNorm(Tensor* X, Tensor* beta, Tensor* gamma,
                      Tensor* mean, Tensor* var, Tensor* Y);
    Tensor* batchNormGrad(Tensor* dY, Tensor* gamma, Tensor* mean,
                      Tensor* var, Tensor* X, Tensor* dX);
    void batchNormParamGrads(Tensor* dY, Tensor* dGamma, Tensor* dBeta,
                      Tensor* Y, Tensor* beta, Tensor* gamma);
} // seblaws

#endif //SEANN_2_CUPARALLEL_CUH
