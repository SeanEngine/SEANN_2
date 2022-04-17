//
// Created by DanielSun on 4/17/2022.
//

#include "cuConv.cuh"

#define BM 128
#define BN 128
#define BK 8
#define RM 8
#define RN 8

#define toFloat4R(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

namespace seblas{
    /**
 * @brief The kernel for the convolution with 4D filters
 * @tparam BLOCK_M
 * @tparam BLOCK_N
 * @tparam BLOCK_K
 * @tparam REGIS_M
 * @tparam REGIS_N
 * @param A oc * ic * fh * fw
 * @param B on * ic * ih * iw
 * @param C on * oc * oh * ow
 * @param strideH
 * @param strideW
 * @param padH
 * @param padW
 */
    template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
            const int REGIS_M, const int REGIS_N>
    __global__ void gemmImplicit4D(Tensor* A, Tensor* B, Tensor* C, int strideH, int strideW, int padH, int padW,
            /*NULLABLE*/ Tensor* biases){

        // MatA: OC, IC * FH * FW; MatB: IC * FH * FW, OH * OW; Mat C: OC, OH * OW
        ///insert parameters
        const uint32 M = A->dims.n;
        const uint32 K = A->dims.c * A->dims.h * A->dims.w;
        const uint32 N = C->dims.n * C->dims.h * C->dims.w;

        const uint32 FH = A->dims.h;
        const uint32 FW = A->dims.w;
        const uint32 IC = B->dims.c;
        const uint32 IH = B->dims.h;
        const uint32 IW = B->dims.w;
        const uint32 OH = C->dims.h;
        const uint32 OW = C->dims.w;
        const uint32 OC = C->dims.c;

        ///allocate smems and registers
        //The shared memory tile
        __shared__ float tileA[2][BLOCK_K][BLOCK_M];  //transposed
        __shared__ float tileB[2][BLOCK_K][BLOCK_N];

        float regisA[2][REGIS_M];
        float regisB[2][REGIS_N];
        float regisC[REGIS_M][REGIS_N] = {0};

        const int threadDimX = BLOCK_N / REGIS_N;
        const int threadDimY = BLOCK_M / REGIS_M;
        const int threadCount = threadDimX * threadDimY;
        const int tid = threadIdx.y * threadDimX + threadIdx.x;

        ///register for buffering elements during transporting global to shared mem
        float bufferA[BLOCK_M * BLOCK_K / threadCount] = {0};
        float bufferB[BLOCK_N * BLOCK_K / threadCount] = {0};

        ///prepare configs for reading global
        float* ptrA = A->elements + blockIdx.y * BLOCK_M * K;
        float* ptrB = B->elements;
        const int blockM = blockIdx.y * BLOCK_M;
        const int blockN = blockIdx.x * BLOCK_N;

        const int readThreadPerRowA = BLOCK_K;
        const int readThreadPerRowB = BLOCK_N;

        //the location each thread should be reading relative to smem
        const int readRowA = tid / readThreadPerRowA;
        const int readColA = tid % readThreadPerRowA;

        const int readRowB = tid / readThreadPerRowB;
        const int readColB = tid % readThreadPerRowB;

        //these values are used to determine the amount of rows to jump
        //if there is the need to do read multiple times
        const int readRowStrideA = threadCount / readThreadPerRowA;
        const int readRowStrideB = threadCount / readThreadPerRowB;

        #pragma unroll
        for(int i=0; i<BLOCK_M; i+= readRowStrideA){
            if(blockM + readRowA + i < M && readColA < K){
                tileA[0][readColA][readRowA+i] = ptrA[(readRowA + i)*K + readColA];
            }
        }

        ///this section is modified from its original state to suit the need for implicit gemm
        ///we are using a special mapping to create patches as trajectories of conv filters
        #pragma unroll
        for(int i=0; i<BLOCK_K; i+= readRowStrideB){
            if(readRowB + i< K && blockN + readColB < N){

                //map buffer matrix cords to the 3 dimensional feature cords
                int in = (readColB + blockN) / (OH * OW);
                int oh = ((readColB + blockN) % (OH * OW))/OW;
                int ow = ((readColB + blockN) % (OH * OW))%OW;
                int ic = (readRowB + i)/(FH * FW);
                int fh = ((readRowB + i)%(FH * FW))/FW;
                int fw = ((readRowB + i)%(FH * FW))%FW;
                int ih = oh * strideH - padH + fh;
                int iw = ow * strideW - padW + fw;
                //do memory access
                tileB[0][readRowB+i][readColB] = ih >= 0 && iw >= 0 && ih < IH && iw < IW ?
                                                 ptrB[in * IC * IH * IW + ic * IH * IW + ih * IW + iw] : 0;
            }
        }
        __syncthreads();


        #pragma unroll
        for(int rm = 0; rm < REGIS_M; rm += 4){
            toFloat4R(regisA[0][rm]) = toFloat4R(tileA[0][0][REGIS_M * threadIdx.y + rm]);
        }

        #pragma unroll
        for(int rn = 0; rn < REGIS_N; rn += 4){
            toFloat4R(regisB[0][rn]) = toFloat4R(tileB[0][0][REGIS_N * threadIdx.x + rn]);
        }

        ///main loop
        int writeStageFlag = 1;
        #pragma unroll
        for(int nextTileID = BLOCK_K; nextTileID < K + BLOCK_K; nextTileID+=BLOCK_K) {
            //prefetch
            if (nextTileID < K) {
                #pragma unroll
                for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                    int loadIndex = i / readRowStrideA;
                    bufferA[loadIndex] = blockM + readRowA + i < M && readColA + nextTileID < K ?
                                         ptrA[(readRowA + i) * K + readColA + nextTileID] : 0;
                }

                #pragma unroll
                for (int i = 0; i < BLOCK_K; i += readRowStrideB) {

                    //calculate remapping
                    int loadIndex = i / readRowStrideB;
                    int in = (readColB + blockN) / (OH * OW);
                    int oh = ((readColB + blockN) % (OH * OW))/OW;
                    int ow = ((readColB + blockN) % (OH * OW))%OW;
                    int ic = (readRowB + i + nextTileID)/(FH * FW);
                    int fh = ((readRowB + i + nextTileID)%(FH * FW))/FW;
                    int fw = ((readRowB + i + nextTileID)%(FH * FW))%FW;
                    int ih = oh * strideH - padH + fh;
                    int iw = ow * strideW - padW + fw;

                    //do memory access
                    bufferB[loadIndex] = (readRowB + i + nextTileID < K && blockN + readColB < N) && (ih >= 0 && iw >= 0)
                                         && (ih < IH && iw < IW)? ptrB[in * IC * IH * IW + ic * IH * IW + ih * IW + iw] : 0;
                }
            }

            int nextStageFlag = writeStageFlag ^ 1;

            //compute the part that is already in the registers and load the next segment
            #pragma unroll
            for (int i = 0; i < BLOCK_K - 1; i++) {

                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm += 4) {
                    toFloat4R(regisA[(i + 1) % 2][rm]) = toFloat4R(
                            tileA[nextStageFlag][i + 1][REGIS_M * threadIdx.y + rm]);
                }

                #pragma unroll
                for (int rn = 0; rn < REGIS_N; rn += 4) {
                    toFloat4R(regisB[(i + 1) % 2][rn]) = toFloat4R(
                            tileB[nextStageFlag][i + 1][REGIS_N * threadIdx.x + rn]);
                }

                #pragma unroll
                for (int rm = 0; rm < REGIS_M; rm++) {
                    #pragma unroll
                    for (int rn = 0; rn < REGIS_N; rn++) {
                        regisC[rm][rn] += regisA[i % 2][rm] * regisB[i % 2][rn];
                    }
                }
            }

            //load the data in the register buffers to tiles
            if (nextTileID < K) {
                #pragma unroll
                for (int i = 0; i < BLOCK_M; i += readRowStrideA) {
                    int loadIndex = i / readRowStrideA;
                    tileA[writeStageFlag][readColA][readRowA + i] = bufferA[loadIndex];
                }

                #pragma unroll
                for (int i = 0; i < BLOCK_K; i += readRowStrideB) {
                    int loadIndex = i / readRowStrideB;
                    tileB[writeStageFlag][readRowB + i][readColB] = bufferB[loadIndex];
                }

                __syncthreads();
                writeStageFlag ^= 1;  //switch
            }
            #pragma unroll
            for (int rm = 0; rm < REGIS_M; rm += 4) {
                toFloat4R(regisA[0][rm]) = toFloat4R(
                        tileA[nextStageFlag ^ 1][0][REGIS_M * threadIdx.y + rm]);
            }

            #pragma unroll
            for (int rn = 0; rn < REGIS_N; rn += 4) {
                toFloat4R(regisB[0][rn]) = toFloat4R(
                        tileB[nextStageFlag ^ 1][0][REGIS_N * threadIdx.x + rn]);
            }

            #pragma unroll
            for(int rm = 0; rm < REGIS_M; rm ++){
                #pragma unroll
                for(int rn = 0; rn < REGIS_N; rn ++){
                    regisC[rm][rn] += regisA[1][rm] * regisB[1][rn];
                }
            }
        }
        #pragma unroll
        for(int rm = 0; rm < REGIS_M; rm ++){
            #pragma unroll
            for(int rn = 0; rn < REGIS_N; rn ++){
                if((blockM + threadIdx.y * REGIS_M + rm < M && blockN + threadIdx.x * REGIS_N + rn < N)) {
                    float bias = biases == nullptr ? 0 : biases->elements[blockM + threadIdx.y * REGIS_M + rm];
                    //remapping is needed since the original output matrix would be ( OC, OH * OW * ON )
                    uint32 on = (blockN + threadIdx.x * REGIS_N + rn) / (OH * OW);
                    C->elements[on * OC * OH * OW + (blockM + threadIdx.y * REGIS_M + rm) * (OH * OW)
                                + (blockN + threadIdx.x * REGIS_N + rn)%(OH * OW)] = regisC[rm][rn] + bias;
                }
            }
        }
    }


    Tensor* conv(Tensor *A, Tensor *B, Tensor *C, int strideH, int strideW, int padH, int padW, Tensor* biases) {
        assertConv(A,B,C, strideH, strideW, padH, padW);

        uint32 M = A->dims.n;
        uint32 N = C->dims.h * C->dims.w * C->dims.n;

        dim3 grid = dim3((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block = dim3(BN / RN, BM / RM);

        gemmImplicit4D<BM, BN, BK, RM, RN><<<grid, block>>>(A, B, C, strideH, strideW, padH, padW, biases);
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return C;
    }
}