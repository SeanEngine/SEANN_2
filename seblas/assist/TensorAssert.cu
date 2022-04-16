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
}