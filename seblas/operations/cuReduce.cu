//
// Created by Dylan on 4/29/2022.
//

#include "cuReduce.cuh"
#define toFloat4R(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
#define topOff(a,b) ((a)+(b) - 1)/(b)

namespace seblas{

    //a warp reduction function that reduce values inside a given warp
    __device__ __forceinline__ float warpReduce(float val){
        #pragma unroll
        for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
            val += __shfl_xor_sync(0x1, val, mask);
        }
        return val;
    }

    //supports step size up to 1024 float32
    template <const uint32 BLOCK_WARPS>
    __global__ void reduceD1024(Tensor* A, Tensor* outA, uint32 step){
        uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32 stepID = blockIdx.y;
        uint32 tid = threadIdx.x;

        //warp reduction
        __shared__ float warpCache[BLOCK_WARPS];
        const uint32 warpId = tid / WARP_SIZE;
        const uint32 laneId = tid % WARP_SIZE;
        float sum = idx < step ? A->elements[stepID * step + idx] : 0;
        __syncthreads();

        sum = warpReduce(sum);
        if(laneId==0) warpCache[warpId] = sum;

        __syncthreads();

        if(warpId==0){
            sum = laneId < BLOCK_WARPS ? warpCache[laneId] : 0.0f;
            sum = warpReduce(sum);
            if(laneId==0) outA->elements[stepID] = sum;
        }
    }

    template <const uint32 BLOCK_WARPS>
    __global__ void reduceD(Tensor* A, Tensor* outA, uint32 reduceStep, uint32 procSize){
        uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32 reduceStepID = blockIdx.y;
        uint32 sourceStepID = blockIdx.z;
        uint32 tid = threadIdx.x;

        //warp reduction
        __shared__ float warpCache[BLOCK_WARPS];
        const uint32 warpId = tid / WARP_SIZE;
        const uint32 laneId = tid % WARP_SIZE;
        float sum = idx < reduceStep && reduceStep * reduceStepID + idx < procSize
                ? A->elements[sourceStepID * procSize + reduceStepID * reduceStep + idx] : 0;
        __syncthreads();

        sum = warpReduce(sum);
        if(laneId==0) warpCache[warpId] = sum;

        __syncthreads();

        if(warpId==0){
            sum = laneId < BLOCK_WARPS ? warpCache[laneId] : 0.0f;
            sum = warpReduce(sum);
            if(laneId==0) outA->elements[sourceStepID *
                topOff(procSize, reduceStep) + reduceStepID] = sum;
        }
    }

    template <const uint32 BLOCK_WARPS>
    __global__ void reduce4D4096(Tensor* A, Tensor* outA, uint32 step){
        uint32 idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
        if(idx >= step) return;

        uint32 stepID = blockIdx.y;
        uint32 tid = threadIdx.x;
        //warp reduction
        __shared__ float warpCache[BLOCK_WARPS];
        const uint32 warpId = tid / WARP_SIZE;
        const uint32 laneId = tid % WARP_SIZE;
        float buffer[4] = {0};
        toFloat4R(buffer) = toFloat4R(A->elements[stepID * step + idx]);

        float sum = 0;
        #pragma unroll
        for (float i : buffer){
            sum += i;
        }

        __syncthreads();

        sum = warpReduce(sum);
        if(laneId==0) warpCache[warpId] = sum;

        __syncthreads();

        if(warpId==0){
            sum = laneId < BLOCK_WARPS ? warpCache[laneId] : 0.0f;
            sum = warpReduce(sum);
            if(laneId==0) outA->elements[stepID] = sum;
        }
    }

    Tensor* reduce(Tensor* A, Tensor* out, Tensor* buffer, uint32 step){
        assert(A->dims.size % step == 0);
        assert(out->dims.size == A->dims.size / step);

        if(step < REDUCE_BLOCK * 4 + 1 && step % 4 == 0){
            dim3 grid = dim3(1, A->dims.size / step);
            dim3 block = REDUCE_BLOCK;
            reduce4D4096<REDUCE_WARPS><<<grid, block>>>(A, out, step);
            assertCuda(__FILE__, __LINE__);
            return out;
        }

        if(step < REDUCE_BLOCK + 1){
            dim3 grid = dim3(1, A->dims.size / step);
            dim3 block = REDUCE_BLOCK;
            reduceD1024<REDUCE_WARPS><<<grid, block>>>(A, out, step);
            assertCuda(__FILE__, __LINE__);
            return out;
        }

        assert(buffer != nullptr);
        assert(buffer->dims.size >= topOff(step, REDUCE_BLOCK) * A->dims.size / step);

        uint32 procSize = step;
        uint32 srcStepCount = A->dims.size / step;
        Tensor* src = A;

        while(procSize > 1){
            dim3 grid = dim3(1, topOff(procSize, REDUCE_BLOCK), srcStepCount);
            uint32 block = REDUCE_BLOCK;
            reduceD<REDUCE_BLOCK><<<grid, block>>>(src, buffer, REDUCE_BLOCK, procSize);
            assertCuda(__FILE__, __LINE__);
            src = buffer;
            procSize = topOff(procSize, REDUCE_BLOCK);
        }

        cudaMemcpy(out->elements, buffer->elements,
                   sizeof(float) * srcStepCount, cudaMemcpyDeviceToDevice);
        assertCuda(__FILE__, __LINE__);
        buffer->constFill(0);
        return out;
    }
}