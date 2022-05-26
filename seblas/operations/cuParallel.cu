//
// Created by Dylan on 5/25/2022.
//

#include "cuParallel.cuh"
#define toFloat4R(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
#define BATCH_NORM_BLOCK 128
#define BATCH_NORM_MAX_PARALLEL 96

namespace seblas {
    __global__ void paraAddD(Tensor* A, Tensor* B, Tensor* C){
        const uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= A->dims.size / A->dims.n ) return;

        float bVal = B->elements[idx];
        uint32 offset = (A->dims.size/A->dims.n);
        //add elements in B to every sample in A
        #pragma unroll
        for(uint32 n = 0; n < A->dims.n ; n++){
            float aVal = A->elements[n * offset + idx];
            C->elements[n * offset + idx] = aVal + bVal;
        }
    }

    __global__ void paraAdd4D(Tensor* A, Tensor* B, Tensor* C){
        const uint32 idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
        if(idx >= A->dims.size / A->dims.n ) return;

        float bVals[4] = {0};
        float aVals[4] = {0};
        float cVals[4] = {0};
        toFloat4R(bVals[0]) = toFloat4R(B->elements[idx]);
        //add elements in B to every sample in A
        #pragma unroll
        for(uint32 n = 0; n < A->dims.n ; n++){
            toFloat4R(aVals[0]) = toFloat4R(A->elements[n * (A->dims.size/A->dims.n) + idx]);
            cVals[0] = aVals[0] + bVals[0];
            cVals[1] = aVals[1] + bVals[1];
            cVals[2] = aVals[2] + bVals[2];
            cVals[3] = aVals[3] + bVals[3];
            toFloat4R(C->elements[n * (A->dims.size/A->dims.n) + idx]) = toFloat4R(cVals[0]);
            cVals[0] = 0;
            cVals[1] = 0;
            cVals[2] = 0;
            cVals[3] = 0;
        }
    }

    template<const uint32 BLOCK, const uint32 MAX_PARALLEL>
    __global__ void batchNormD(Tensor* X, Tensor* beta, Tensor* gamma,
                               Tensor* mean, Tensor* var, Tensor* Y){
        const uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= X->dims.size /X->dims.n ) return;

        //since each SM has access to 163KB of shared memory
        __shared__ float xs[BLOCK * MAX_PARALLEL];
        float meanVal;
        float varVal;

        //load data into shared memory
        #pragma unroll
        for(uint32 depth = 0; depth < X->dims.n; depth++){
            float xVal = X->elements[depth * (X->dims.size/X->dims.n) + idx];
            xs[depth * BLOCK + threadIdx.x] = xVal;
            meanVal += xVal;
        }
        meanVal /= (float)X->dims.n;

        //compute variance
        #pragma unroll
        for(uint32 depth = 0; depth < X->dims.n; depth++){
            float xVal = xs[depth * BLOCK + threadIdx.x];
            varVal += (xVal - meanVal) * (xVal -meanVal);
        }
        varVal /= (float)X->dims.n;

        //compute xHat
        #pragma unroll
        for(uint32 depth = 0; depth < X->dims.n; depth++){
            float xVal = xs[depth * BLOCK + threadIdx.x];
            xs[depth * BLOCK + threadIdx.x] = (xVal - meanVal) / sqrt(varVal + 1e-8);
        }

        //apply transformation
        #pragma unroll
        for(uint32 depth = 0; depth < X->dims.n; depth++){
            float xVal = xs[depth * BLOCK + threadIdx.x];
            float betaVal = beta->elements[idx];
            float gammaVal = gamma->elements[idx];
            Y->elements[depth * (Y->dims.size / Y->dims.n) + idx] = xVal * gammaVal + betaVal;
        }

        mean->elements[idx] = meanVal;
        var->elements[idx] = varVal;
    }

    template<const uint32 BLOCK, const uint32 MAX_PARALLEL>
    __global__ void batchNorm4D(Tensor* X, Tensor* beta, Tensor* gamma,
                                Tensor* mean, Tensor* var, Tensor* Y){
        const uint32 idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
        const uint32 threadId = threadIdx.x * 4;
        if(idx >= X->dims.size /X->dims.n ) return;

        //since each SM has access to 163KB of shared memory
        __shared__ float xs[BLOCK * MAX_PARALLEL * 4];

        float xVal[4];
        float meanVal[4];
        float varVal[4];

        //load x and compute mean
        #pragma unroll
        for(uint32 depth = 0; depth < X->dims.n; depth++){
            toFloat4R(xVal[0]) = toFloat4R(X->elements[depth * (X->dims.size/X->dims.n) + idx]);
            toFloat4R(xs[depth * BLOCK * 4 + threadId]) = toFloat4R(xVal[0]);
            meanVal[0] += xVal[0];
            meanVal[1] += xVal[1];
            meanVal[2] += xVal[2];
            meanVal[3] += xVal[3];
        }

        meanVal[0] /= (float)X->dims.n;
        meanVal[1] /= (float)X->dims.n;
        meanVal[2] /= (float)X->dims.n;
        meanVal[3] /= (float)X->dims.n;

        //compute variance
        #pragma unroll
        for(uint32 depth = 0; depth < X->dims.n; depth++) {
            toFloat4R(xVal[0]) = toFloat4R(xs[depth * BLOCK * 4 + threadId]);
            varVal[0] += (xVal[0] - meanVal[0]) * (xVal[0] -meanVal[0]);
            varVal[1] += (xVal[1] - meanVal[1]) * (xVal[1] -meanVal[1]);
            varVal[2] += (xVal[2] - meanVal[2]) * (xVal[2] -meanVal[2]);
            varVal[3] += (xVal[3] - meanVal[3]) * (xVal[3] -meanVal[3]);
        }

        varVal[0] /= (float)X->dims.n;
        varVal[1] /= (float)X->dims.n;
        varVal[2] /= (float)X->dims.n;
        varVal[3] /= (float)X->dims.n;

        //compute xHat
        #pragma unroll
        for(uint32 depth = 0; depth < X->dims.n; depth++){
            toFloat4R(xVal[0]) = toFloat4R(xs[depth * BLOCK * 4 + threadId]);
            xVal[0] = (xVal[0] - meanVal[0]) / (float)sqrt(varVal[0] + 1e-8);
            xVal[1] = (xVal[1] - meanVal[1]) / (float)sqrt(varVal[1] + 1e-8);
            xVal[2] = (xVal[2] - meanVal[2]) / (float)sqrt(varVal[2] + 1e-8);
            xVal[3] = (xVal[3] - meanVal[3]) / (float)sqrt(varVal[3] + 1e-8);
            toFloat4R(xs[depth * BLOCK * 4 + threadId]) = toFloat4R(xVal[0]);
        }

        float betaVal[4];
        float gammaVal[4];

        //apply transformation
        #pragma unroll
        for(uint32 depth = 0; depth < X->dims.n; depth++){
            toFloat4R(xVal[0]) = toFloat4R(xs[depth * BLOCK * 4 + threadId]);
            toFloat4R(betaVal[0]) = toFloat4R(beta->elements[idx]);
            toFloat4R(gammaVal[0]) = toFloat4R(gamma->elements[idx]);
            xVal[0] = gammaVal[0] * xVal[0] + betaVal[0];
            xVal[1] = gammaVal[1] * xVal[1] + betaVal[1];
            xVal[2] = gammaVal[2] * xVal[2] + betaVal[2];
            xVal[3] = gammaVal[3] * xVal[3] + betaVal[3];
            toFloat4R(Y->elements[depth * (Y->dims.size/Y->dims.n) + idx]) = toFloat4R(xVal[0]);
        }

        toFloat4R(mean->elements[idx]) = toFloat4R(meanVal[0]);
        toFloat4R(var->elements[idx]) = toFloat4R(varVal[0]);
    }

    template<const uint32 BLOCK, const uint32 MAX_PARALLEL>
    __global__ void batchNormGradD(Tensor* dY, Tensor* gamma,
                                   Tensor* mean, Tensor* var, Tensor* X, Tensor* dX){
        const uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= dY->dims.size /dY->dims.n ) return;

        __shared__ float xs[BLOCK * MAX_PARALLEL];

        float meanVal = mean->elements[idx];
        float varVal = var->elements[idx];
        float dVar = 0;
        float dMean = 0;
        float temp = 0;   // A part of dMean

        //calculate dX
        #pragma unroll
        for(uint32 depth = 0; depth < dY->dims.n; depth++){
            float xVal = X->elements[depth * (X->dims.size/X->dims.n) + idx];
            xs[depth * BLOCK + threadIdx.x] = xVal;
            float gammaVal = gamma->elements[idx];

            //calculate dXHat
            float dxHat = dY->elements[depth * (dY->dims.size/dY->dims.n) + idx] * gammaVal;
            dVar += dxHat * (xVal - meanVal) * -0.5f * pow(varVal + 1e-8f, -3.0f/2.0f);
            dMean += dxHat * -1.0f / sqrt(varVal + 1e-8f);
            temp += -2.0f * (xVal - meanVal);
        }

        //calculate dMean
        dMean += dVar * temp / (float)dY->dims.n;

        //calculate dX
        #pragma unroll
        for(uint32 depth = 0; depth < dY->dims.n; depth++){
            float xVal = xs[depth * BLOCK + threadIdx.x];
            float dy = dY->elements[depth * (dY->dims.size/dY->dims.n) + idx];
            float gammaVal = gamma->elements[idx];
            float dXHat = dy * gammaVal;
            dX->elements[depth * (dX->dims.size/dX->dims.n) + idx] = dXHat / sqrt(varVal + 1e-8f)
                    + dVar * (xVal - meanVal) * 2.0f / (float)dY->dims.n + dMean / (float)dY->dims.n;
        }
    }

    __global__ void batchNormParamGradsD(Tensor* dY, Tensor* dGamma, Tensor* dBeta,
                                         Tensor* Y, Tensor* beta, Tensor* gamma){
        const uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;

        float dBetaVal = 0;
        float dGammaVal = 0;

        #pragma unroll
        for(uint32 depth = 0; depth < dY->dims.n; depth++){
            float dy = dY->elements[depth * (dY->dims.size/dY->dims.n) + idx];
            float yVal = Y->elements[depth * (Y->dims.size/Y->dims.n) + idx];
            float betaVal = beta->elements[idx];
            float gammaVal = gamma->elements[idx];

            float xHat = gammaVal == 0 ? 0 : (yVal - betaVal) / (gammaVal);

            dBetaVal += dy;
            dGammaVal += dy * xHat;
        }

        dBeta->elements[idx] = dBetaVal;
        dGamma->elements[idx] = dGammaVal;
    }

    Tensor* paraAdd(Tensor *A, Tensor *B, Tensor *C) {
        assert(A->dims.size == C->dims.size);
        assert(B->dims.size == C->dims.size / C->dims.n);

        uint32 block = CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y;
        uint32 grid = ((A->dims.size / A->dims.n) + block - 1) / block;
        if((A->dims.size / A->dims.n) % 4 == 0) {
            grid = ((A->dims.size / A->dims.n) + block * 4 - 1) / (block * 4);
            paraAdd4D<<<grid, block>>>(A, B, C);
            cudaDeviceSynchronize();
            assertCuda(__FILE__, __LINE__);
            return C;
        }
        paraAddD<<<grid, block>>>(A, B, C);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return C;
    }

    Tensor* paraAdd(Tensor* A, Tensor* B){
        return paraAdd(A, B, A);
    }

    Tensor* batchNorm(Tensor* X, Tensor* beta, Tensor* gamma,
                      Tensor* mean, Tensor* var, Tensor* Y){
        assert(X->dims.n == Y->dims.n);
        assert(X->dims.size == Y->dims.size);

        uint32 block = BATCH_NORM_BLOCK;
        uint32 grid = ((X->dims.size / X->dims.n) + block - 1) / block;

        if((X->dims.size / X->dims.n) % 4 == 0) {
            grid = ((X->dims.size / X->dims.n) + block  - 1) / (block );
            batchNorm4D<BATCH_NORM_BLOCK/4, BATCH_NORM_MAX_PARALLEL>
                    <<<grid, block/4>>>(X, beta, gamma, mean, var, Y);
            cudaDeviceSynchronize();
            assertCuda(__FILE__, __LINE__);
            return Y;
        }

        batchNormD<BATCH_NORM_BLOCK, BATCH_NORM_MAX_PARALLEL>
                <<<grid, block>>>(X, beta, gamma, mean, var, Y);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return Y;
    }
} // seblaws