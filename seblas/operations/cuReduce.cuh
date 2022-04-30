//
// Created by Dylan on 4/29/2022.
//

#ifndef SEANN_2_CUREDUCE_CUH
#define SEANN_2_CUREDUCE_CUH

#include "../tensor/Tensor.cuh"

namespace seblas{

    const uint32 REDUCE_BLOCK = 1024;
    const uint32 REDUCE_WARPS = 32;

    /**
     * @brief Reduce the given tensor with steps.
     *
     * @param A input tensor
     * @param out output tensor with shape (A->size / steps, 1)
     * @param step the size of chunk to be summed up
     * @param buffer buffer for reduction (can be nullPointer)
     * @return
     */
    Tensor* reduce(Tensor* A, Tensor* out, Tensor* buffer, uint32 step);

    //column reduce : stride = w
    Tensor* reduce(Tensor* A, Tensor* out, uint32 step, uint32 stride);

    float reduce(Tensor* A);
}

#endif //SEANN_2_CUREDUCE_CUH
