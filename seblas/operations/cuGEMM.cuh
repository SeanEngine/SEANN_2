//
// Created by DanielSun on 4/17/2022.
//

#ifndef SEANN_2_CUGEMM_CUH
#define SEANN_2_CUGEMM_CUH

#include "../tensor/Tensor.cuh"

namespace seblas{
    Tensor* sgemm(Tensor *A, Tensor *B, Tensor *C);

    Tensor* sgemmTN(Tensor *A, Tensor *B, Tensor *C);

    Tensor* sgemmNT(Tensor *A, Tensor *B, Tensor *C);

    Tensor* sgemmNTA(Tensor *A, Tensor *B, Tensor *C);
}

#endif //SEANN_2_CUGEMM_CUH
