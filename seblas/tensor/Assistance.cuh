//
// Created by Dylan on 4/30/2022.
//

#ifndef SEANN_2_ASSISTANCE_CUH
#define SEANN_2_ASSISTANCE_CUH

#include "Tensor.cuh"

namespace seblas{
    Tensor* transpose(Tensor* A);
    Tensor* transpose(Tensor* A, Tensor* B);
}

#endif //SEANN_2_ASSISTANCE_CUH
