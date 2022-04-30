//
// Created by Dylan on 4/30/2022.
//

#include "Assistance.cuh"
#define topOff(a,b) (((a)+(b)-1)/(b))

namespace seblas{
    __global__ void transposeD(Tensor* A, Tensor* B){
        uint32 row = blockIdx.x * blockDim.x + threadIdx.x;
        uint32 col = blockIdx.y * blockDim.y + threadIdx.y;
        uint32 channel = blockIdx.z * blockDim.z + threadIdx.z;

        if(row < A->dims.h && col < A->dims.w && channel < A->dims.c * A->dims.n){
            B->elements[channel * A->dims.w * A->dims.h + col * A->dims.h + row]
                = A->elements[channel * A->dims.w * A->dims.h + row * A->dims.w + col];
        }
    }

    Tensor* transpose(Tensor* A, Tensor* B){
        assert(A->dims.size == B->dims.size);
        dim3 block = CUDA_BLOCK_SIZE_3D;
        dim3 grid = {topOff(A->dims.w, block.x),
                     topOff(A->dims.h, block.y),
                     topOff(A->dims.c * A->dims.n, block.z)};
        transposeD<<<grid, block>>>(A, B);
        assertCuda(__FILE__, __LINE__);
        return B;
    }

    Tensor* transpose(Tensor* A){
        return transpose(A,A)->reshape(A->dims.n, A->dims.c, A->dims.w, A->dims.h);
    }
}