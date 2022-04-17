//
// Created by Wake on 4/16/2022.
//

#include "TensorAssert.cuh"

namespace seblas{
    void assertInRange(Tensor* A, Tensor* B){
        if(A->dims.size >= B->dims.size){
            logFatal(seio::LOG_SEG_SEBLAS, "Tensor assert failed:");
            throw std::invalid_argument("assertInRange: A exceeds range of B");
        }
    }

    void assertInRangeStrict(Tensor* A, Tensor* B){
       if(!(A->dims < B->dims)){
           logFatal(seio::LOG_SEG_SEBLAS, "Tensor assert failed:");
           throw std::invalid_argument("assertInRangeStrict: A exceeds range of B");
       }
    }

    void assertConv(Tensor* A, Tensor* B, Tensor* C, uint32 strideH, uint32 strideW, uint32 padH, uint32 padW){
        if(C->dims.h != (B->dims.h - A->dims.h + 2 * padH) / strideH + 1){
            logFatal(seio::LOG_SEG_SEBLAS, "Tensor assert failed:");
            throw std::invalid_argument("assertConv: rows (h) relationship did not satisfy");
        }

        if(C->dims.w != (B->dims.w - A->dims.w + 2 * padW) / strideW + 1){
            logFatal(seio::LOG_SEG_SEBLAS, "Tensor assert failed:");
            throw std::invalid_argument("assertConv: cols (w) relationship did not satisfy");
        }

        if(C->dims.c != A->dims.n || B->dims.c != A->dims.c){
            logFatal(seio::LOG_SEG_SEBLAS, "Tensor assert failed:");
            throw std::invalid_argument("assertConv: channels relationship did not satisfy");
        }

        if(C->dims.n != B->dims.n){
            logFatal(seio::LOG_SEG_SEBLAS, "Tensor assert failed:");
            throw std::invalid_argument("assertConv: batch relationship did not satisfy");
        }
    }
}