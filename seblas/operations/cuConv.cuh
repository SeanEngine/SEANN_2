//
// Created by DanielSun on 4/17/2022.
//

#ifndef SEANN_2_CUCONV_CUH
#define SEANN_2_CUCONV_CUH

#include "../tensor/Tensor.cuh"
#include "../assist/TensorAssert.cuh"

namespace seblas{
    /**
     * Convolution operator by GEMM (will implement FFT and vinograd in future)
     * @param A filters OC * IC * FH * FW
     * @param B feature inputs N * IC * IH * IW
     * @param C feature outputs N ＊ OC * OH * OW
     * @param biases set this to be nullpointer if you don't want biases (OC * 1)
     * @return
     */
    Tensor* conv(Tensor *A, Tensor *B, Tensor *C, int strideH, int strideW, int padH, int padW, Tensor* biases);
}

#endif //SEANN_2_CUCONV_CUH
