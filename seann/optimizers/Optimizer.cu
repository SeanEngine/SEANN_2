//
// Created by Dylan on 5/1/2022.
//

#include "Optimizer.cuh"
#define toFloat4R(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
#define topOff(a,b) (((a)+(b)-1)/(b))

namespace seann {
    __global__ void SGDApplyD(Parameter* A, float LEARNING_RATE){
        uint32 idx = threadIdx.x + blockDim.x * blockIdx.x;
        if(idx < A->a->dims.size){
            A->a->elements[idx] -= LEARNING_RATE * A->grad->elements[idx];
            A->grad->elements[idx] = 0;
        }
    }

    __global__ void SGDApply4D(Parameter* A, float LEARNING_RATE){
        uint32 idx = (threadIdx.x + blockDim.x * blockIdx.x) * 4;
        float regisA[4] = {0};
        float regisG[4] = {0};
        if(idx < A->a->dims.size){
            toFloat4R(regisA[0]) = toFloat4R(A->a->elements[idx]);
            toFloat4R(regisG[0]) = toFloat4R(A->grad->elements[idx]);
            regisA[0] -= LEARNING_RATE * regisG[0];
            regisA[1] -= LEARNING_RATE * regisG[1];
            regisA[2] -= LEARNING_RATE * regisG[2];
            regisA[3] -= LEARNING_RATE * regisG[3];
            regisG[0] = 0;
            regisG[1] = 0;
            regisG[2] = 0;
            regisG[3] = 0;
            toFloat4R(A->a->elements[idx]) = toFloat4R(regisA[0]);
            toFloat4R(A->grad->elements[idx]) = toFloat4R(regisG[0]);
        }
    }

    __global__ void momentumApplyD(Parameter* A, Tensor* m, float LEARNING_RATE, float BETA){
        uint32 idx = threadIdx.x + blockDim.x * blockIdx.x;
        if(idx < A->a->dims.size){
            float mVal = BETA * m->elements[idx] + (1-BETA) * A->grad->elements[idx];
            A->a->elements[idx] -= LEARNING_RATE * mVal;
            m->elements[idx] = mVal;
            A->grad->elements[idx] = 0;
        }
    }

    __global__ void momentumApply4D(Parameter* A, Tensor* m, float LEARNING_RATE, float BETA){
        uint32 idx = (threadIdx.x + blockDim.x * blockIdx.x) * 4;
        float regisA[4] = {0};
        float regisG[4] = {0};
        float regisM[4] = {0};
        if(idx < A->a->dims.size){
            toFloat4R(regisA[0]) = toFloat4R(A->a->elements[idx]);
            toFloat4R(regisG[0]) = toFloat4R(A->grad->elements[idx]);
            toFloat4R(regisM[0]) = toFloat4R(m->elements[idx]);
            regisM[0] = BETA * regisM[0] + (1-BETA) * regisG[0];
            regisM[1] = BETA * regisM[1] + (1-BETA) * regisG[1];
            regisM[2] = BETA * regisM[2] + (1-BETA) * regisG[2];
            regisM[3] = BETA * regisM[3] + (1-BETA) * regisG[3];
            regisA[0] -= LEARNING_RATE * regisM[0];
            regisA[1] -= LEARNING_RATE * regisM[1];
            regisA[2] -= LEARNING_RATE * regisM[2];
            regisA[3] -= LEARNING_RATE * regisM[3];
            regisG[0] = 0;
            regisG[1] = 0;
            regisG[2] = 0;
            regisG[3] = 0;
            toFloat4R(A->a->elements[idx]) = toFloat4R(regisA[0]);
            toFloat4R(m->elements[idx]) = toFloat4R(regisM[0]);
            toFloat4R(A->grad->elements[idx]) = toFloat4R(regisG[0]);
        }
    }

    __global__ void adaGradApplyD(Parameter* A, Tensor* V, float LEARNING_RATE, float EPSILON){
        uint32 idx = threadIdx.x + blockDim.x * blockIdx.x;
        if(idx < A->a->dims.size){
            float gradVal = A->grad->elements[idx];
            float vVal = V->elements[idx] + gradVal * gradVal;
            A->a->elements[idx] -= LEARNING_RATE * gradVal / (sqrt(vVal) + EPSILON);
            V->elements[idx] = vVal;
            A->grad->elements[idx] = 0;
        }
    }

    __global__ void adaGradApply4D(Parameter* A, Tensor* V, float LEARNING_RATE, float EPSILON) {
        uint32 idx = (threadIdx.x + blockDim.x * blockIdx.x) * 4;
        float regisA[4] = {0};
        float regisG[4] = {0};
        float regisV[4] = {0};
        if (idx < A->a->dims.size) {
            toFloat4R(regisA[0]) = toFloat4R(A->a->elements[idx]);
            toFloat4R(regisG[0]) = toFloat4R(A->grad->elements[idx]);
            toFloat4R(regisV[0]) = toFloat4R(V->elements[idx]);
            regisV[0] += regisG[0] * regisG[0];
            regisV[1] += regisG[1] * regisG[1];
            regisV[2] += regisG[2] * regisG[2];
            regisV[3] += regisG[3] * regisG[3];
            regisA[0] -= LEARNING_RATE * regisG[0] / (sqrt(regisV[0]) + EPSILON);
            regisA[1] -= LEARNING_RATE * regisG[1] / (sqrt(regisV[1]) + EPSILON);
            regisA[2] -= LEARNING_RATE * regisG[2] / (sqrt(regisV[2]) + EPSILON);
            regisA[3] -= LEARNING_RATE * regisG[3] / (sqrt(regisV[3]) + EPSILON);
            regisG[0] = 0;
            regisG[1] = 0;
            regisG[2] = 0;
            regisG[3] = 0;
            toFloat4R(A->a->elements[idx]) = toFloat4R(regisA[0]);
            toFloat4R(V->elements[idx]) = toFloat4R(regisV[0]);
            toFloat4R(A->grad->elements[idx]) = toFloat4R(regisG[0]);
        }
    }

    __global__ void adaDeltaApplyD(Parameter* A, Tensor* V, Tensor* Vx, float EPSILON, float BETA){
        uint32 idx = threadIdx.x + blockDim.x * blockIdx.x;
        if(idx < A->a->dims.size){
            float gradVal = A->grad->elements[idx];
            float prevUpdate = Vx->elements[idx];
            float vVal = BETA * V->elements[idx] + (1-BETA) * gradVal * gradVal;
            float update = - gradVal * sqrt(prevUpdate + EPSILON) / sqrt(vVal + EPSILON);
            float accu = BETA * prevUpdate + (1-BETA) * update * update;

            A->a->elements[idx] += update;
            V->elements[idx] = vVal;
            Vx->elements[idx] = accu;
            A->grad->elements[idx] = 0;
        }
    }

    __global__ void adaDeltaApply4D(Parameter* A, Tensor* V, Tensor* Vx, float EPSILON, float BETA){
        uint32 idx = (threadIdx.x + blockDim.x * blockIdx.x) * 4;
        float regisA[4] = {0};
        float regisG[4] = {0};
        float regisV[4] = {0};
        float regisVx[4] = {0};
        float update[4] = {0};
        if (idx < A->a->dims.size) {
            toFloat4R(regisA[0]) = toFloat4R(A->a->elements[idx]);
            toFloat4R(regisG[0]) = toFloat4R(A->grad->elements[idx]);
            toFloat4R(regisV[0]) = toFloat4R(V->elements[idx]);
            toFloat4R(regisVx[0]) = toFloat4R(Vx->elements[idx]);

            regisV[0] = BETA * regisV[0] + (1-BETA) * regisG[0] * regisG[0];
            regisV[1] = BETA * regisV[1] + (1-BETA) * regisG[1] * regisG[1];
            regisV[2] = BETA * regisV[2] + (1-BETA) * regisG[2] * regisG[2];
            regisV[3] = BETA * regisV[3] + (1-BETA) * regisG[3] * regisG[3];

            update[0] = - regisG[0] * sqrt(regisVx[0] + EPSILON) / sqrt(regisV[0] + EPSILON);
            update[1] = - regisG[1] * sqrt(regisVx[1] + EPSILON) / sqrt(regisV[1] + EPSILON);
            update[2] = - regisG[2] * sqrt(regisVx[2] + EPSILON) / sqrt(regisV[2] + EPSILON);
            update[3] = - regisG[3] * sqrt(regisVx[3] + EPSILON) / sqrt(regisV[3] + EPSILON);

            regisVx[0] = BETA * regisVx[0] + (1-BETA) * update[0] * update[0];
            regisVx[1] = BETA * regisVx[1] + (1-BETA) * update[1] * update[1];
            regisVx[2] = BETA * regisVx[2] + (1-BETA) * update[2] * update[2];
            regisVx[3] = BETA * regisVx[3] + (1-BETA) * update[3] * update[3];

            regisA[0] += update[0];
            regisA[1] += update[1];
            regisA[2] += update[2];
            regisA[3] += update[3];

            regisG[0] = 0;
            regisG[1] = 0;
            regisG[2] = 0;
            regisG[3] = 0;

            toFloat4R(A->a->elements[idx]) = toFloat4R(regisA[0]);
            toFloat4R(V->elements[idx]) = toFloat4R(regisV[0]);
            toFloat4R(Vx->elements[idx]) = toFloat4R(regisVx[0]);
            toFloat4R(A->grad->elements[idx]) = toFloat4R(regisG[0]);

        }
    }

    __global__ void adamApplyD(Parameter* A, Tensor* m, Tensor* V, float LEARNING_RATE, float EPSILON, float BETA1, float BETA2){
        uint32 idx = threadIdx.x + blockDim.x * blockIdx.x;
        if(idx < A->a->dims.size){
            float gradVal = A->grad->elements[idx];
            float mVal = BETA1 * m->elements[idx] + (1-BETA1) * gradVal;
            float vVal = BETA2 * V->elements[idx] + (1-BETA2) * gradVal * gradVal;
            A->a->elements[idx] -= LEARNING_RATE * (mVal / (sqrt(vVal) + EPSILON));
            m->elements[idx] = mVal;
            V->elements[idx] = vVal;
            A->grad->elements[idx] = 0;
        }
    }

    __global__ void adamApply4D(Parameter* A, Tensor* m, Tensor* V, float LEARNING_RATE, float EPSILON, float BETA1, float BETA2){
        uint32 idx = (threadIdx.x + blockDim.x * blockIdx.x) * 4;
        float regisA[4] = {0};
        float regisG[4] = {0};
        float regisM[4] = {0};
        float regisV[4] = {0};
        if (idx < A->a->dims.size) {
            toFloat4R(regisA[0]) = toFloat4R(A->a->elements[idx]);
            toFloat4R(regisG[0]) = toFloat4R(A->grad->elements[idx]);
            toFloat4R(regisM[0]) = toFloat4R(m->elements[idx]);
            toFloat4R(regisV[0]) = toFloat4R(V->elements[idx]);
            regisV[0] = BETA2 * regisV[0] + (1-BETA2) * regisG[0] * regisG[0];
            regisV[1] = BETA2 * regisV[1] + (1-BETA2) * regisG[1] * regisG[1];
            regisV[2] = BETA2 * regisV[2] + (1-BETA2) * regisG[2] * regisG[2];
            regisV[3] = BETA2 * regisV[3] + (1-BETA2) * regisG[3] * regisG[3];
            regisM[0] = BETA1 * regisM[0] + (1-BETA1) * regisG[0];
            regisM[1] = BETA1 * regisM[1] + (1-BETA1) * regisG[1];
            regisM[2] = BETA1 * regisM[2] + (1-BETA1) * regisG[2];
            regisM[3] = BETA1 * regisM[3] + (1-BETA1) * regisG[3];
            regisA[0] -= LEARNING_RATE * regisM[0] / (sqrt(regisV[0]) + EPSILON);
            regisA[1] -= LEARNING_RATE * regisM[1] / (sqrt(regisV[1]) + EPSILON);
            regisA[2] -= LEARNING_RATE * regisM[2] / (sqrt(regisV[2]) + EPSILON);
            regisA[3] -= LEARNING_RATE * regisM[3] / (sqrt(regisV[3]) + EPSILON);
            toFloat4R(A->a->elements[idx]) = toFloat4R(regisA[0]);
            toFloat4R(m->elements[idx]) = toFloat4R(regisM[0]);
            toFloat4R(V->elements[idx]) = toFloat4R(regisV[0]);
            toFloat4R(A->grad->elements[idx]) = toFloat4R(regisG[0]);
        }
    }

    void SGD::apply() {
        uint32 block = CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y;
        uint32 grid = topOff(A->a->dims.size, block);
        if(A->a->dims.size % 4 == 0){
            grid = topOff(A->a->dims.size, block * 4);
            SGDApply4D<<<grid, block>>>(A, LEARNING_RATE);
            assertCuda(__FILE__, __LINE__);
            return;
        }
        SGDApplyD<<<grid, block>>>(A, LEARNING_RATE);
        assertCuda(__FILE__, __LINE__);
    }

    void BGD::batchApply() {
        uint32 block = CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y;
        uint32 grid = topOff(A->a->dims.size, block);
        if(A->a->dims.size % 4 == 0){
            grid = topOff(A->a->dims.size, block * 4);
            SGDApply4D<<<grid, block>>>(A, LEARNING_RATE/BATCH_SIZE);
            assertCuda(__FILE__, __LINE__);
            return;
        }
        SGDApplyD<<<grid, block>>>(A, LEARNING_RATE/BATCH_SIZE);
        assertCuda(__FILE__, __LINE__);
    }

    void Momentum::apply() {
        uint32 block = CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y;
        uint32 grid = topOff(A->a->dims.size, block);
        if(A->a->dims.size % 4 == 0){
            grid = topOff(A->a->dims.size, block * 4);
            momentumApply4D<<<grid, block>>>(A, m, LEARNING_RATE, BETA);
            assertCuda(__FILE__, __LINE__);
            return;
        }
        momentumApplyD<<<grid, block>>>(A, m, LEARNING_RATE, BETA);
        assertCuda(__FILE__, __LINE__);
    }

    void AdaGrad::apply() {
        uint32 block = CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y;
        uint32 grid = topOff(A->a->dims.size, block);
        if(A->a->dims.size % 4 == 0){
            grid = topOff(A->a->dims.size, block * 4);
            adaGradApply4D<<<grid, block>>>(A, V, LEARNING_RATE, EPSILON);
            assertCuda(__FILE__, __LINE__);
            return;
        }
        adaGradApplyD<<<grid, block>>>(A, V, LEARNING_RATE, EPSILON);
        assertCuda(__FILE__, __LINE__);
    }

    void AdaDelta::apply() {
        uint32 block = CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y;
        uint32 grid = topOff(A->a->dims.size, block);

        if(A->a->dims.size % 4 == 0){
            grid = topOff(A->a->dims.size, block * 4);
            adaDeltaApply4D<<<grid, block>>>(A,  V, Vx, EPSILON, BETA);
            assertCuda(__FILE__, __LINE__);
            return;
        }

        adaDeltaApplyD<<<grid, block>>>(A,  V, Vx, EPSILON, BETA);
        assertCuda(__FILE__, __LINE__);
    }

    void Adam::apply() {
        uint32 block = CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y;
        uint32 grid = topOff(A->a->dims.size, block);
        if(A->a->dims.size % 4 == 0){
            grid = topOff(A->a->dims.size, block * 4);
            adamApply4D<<<grid, block>>>(A, m, V, LEARNING_RATE, EPSILON, BETA1, BETA2);
            assertCuda(__FILE__, __LINE__);
            return;
        }
        adamApplyD<<<grid, block>>>(A, m, V, LEARNING_RATE, EPSILON, BETA1, BETA2);
        assertCuda(__FILE__, __LINE__);
    }
} // seann