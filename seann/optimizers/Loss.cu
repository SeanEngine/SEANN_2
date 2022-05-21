//
// Created by Dylan on 5/13/2022.
//
#include "Loss.cuh"

namespace seann{

    __global__ void crossEntropyPrepare(Parameter* Y, Tensor* label, Tensor* buf){
        uint32 idx = threadIdx.x + blockIdx.x * blockDim.x;
        if(idx < Y->a->dims.size){
            float s = Y->a->elements[idx];
            buf->elements[idx] = label->elements[idx] * (-log(s + 1e-10f));
        }
    }

    float crossEntropyCalc(Parameter* Y, Tensor* label, Tensor* buf){
        uint32 block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        uint32 grid = (label->dims.size + block - 1) / block;

        crossEntropyPrepare<<<grid, block>>>(Y, label, buf);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);

        float out =  reduce(buf, buf);
        buf->constFill(0.0f);
        return out;
    }

    void crossEntropyLoss(Parameter* Y, Tensor* label){
        Y->a->copyToD2D(Y->grad);
        *Y->grad - label;
    }
}