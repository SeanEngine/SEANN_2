//
// Created by Wake on 4/16/2022.
//

#ifndef SEANN_2_TENSORASSERT_CUH
#define SEANN_2_TENSORASSERT_CUH

#include "../tensor/Tensor.cuh"
#include "../../seio/logging/LogUtils.cuh"

using namespace seio;
namespace seblas {
    //assert whether the size of A is within B
    void assertInRange(Tensor* A, Tensor* B);

    //assert strict dimension relationship
    void assertInRangeStrict(Tensor* A, Tensor* B);
}


#endif //SEANN_2_TENSORASSERT_CUH
