//
// Created by Dylan on 5/13/2022.
//
#include "Loss.cuh"

namespace seann{

    __global__ void crossEntropyPrepare(Parameter* Y, Tensor* labels){
        uint32 idx = threadIdx.x + blockIdx.x * blockDim.x;
        if(idx < Y->a->dims.size ){
            float labelVal = labels->elements[idx];
            labels->elements[idx] = -log(Y->a->elements[idx] + 1e-10f) * labelVal;
        }
    }

    float crossEntropyCalc(Parameter* Y, Tensor* buf){
        uint32 block = CUDA_BLOCK_SIZE.y * CUDA_BLOCK_SIZE.x;
        uint32 grid = (buf->dims.size + block - 1) / block;

        crossEntropyPrepare<<<grid, block>>>(Y, buf);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);

        float out =  reduce(buf, buf);
        buf->constFill(0.0f);
        return out;
    }

    void crossEntropyLoss(Parameter* Y, Tensor* labels){
        Y->a->copyToD2D(Y->grad);
        *Y->grad - labels;
    }
}