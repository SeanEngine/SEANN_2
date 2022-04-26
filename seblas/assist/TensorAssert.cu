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

    void assertConv(Tensor* filters, Tensor* features, Tensor* featureOut, uint32 strideH, uint32 strideW, uint32 padH, uint32 padW){
        if(featureOut->dims.h != (features->dims.h - filters->dims.h + 2 * padH) / strideH + 1){
            logFatal(seio::LOG_SEG_SEBLAS, "Tensor assert failed:");
            logFatal(seio::LOG_SEG_SEBLAS, "assertConv: rows (h) relationship did not satisfy");
            throw std::invalid_argument("assertConv: rows (h) relationship did not satisfy");
        }

        if(featureOut->dims.w != (features->dims.w - filters->dims.w + 2 * padW) / strideW + 1){
            logFatal(seio::LOG_SEG_SEBLAS, "Tensor assert failed:");
            logFatal(seio::LOG_SEG_SEBLAS, "assertConv: cols (w) relationship did not satisfy");
            throw std::invalid_argument("assertConv: cols (w) relationship did not satisfy");
        }

        if(featureOut->dims.c != filters->dims.n || features->dims.c != filters->dims.c){
            logFatal(seio::LOG_SEG_SEBLAS, "Tensor assert failed:");
            logFatal(seio::LOG_SEG_SEBLAS, "assertConv: channels (c) relationship did not satisfy");
            throw std::invalid_argument("assertConv: channels relationship did not satisfy");
        }

        if(featureOut->dims.n != features->dims.n){
            logFatal(seio::LOG_SEG_SEBLAS, "Tensor assert failed:");
            logFatal(seio::LOG_SEG_SEBLAS, "assertConv: batch (n) relationship did not satisfy");
            throw std::invalid_argument("assertConv: batch relationship did not satisfy");
        }
    }
}