//
// Created by DanielSun on 4/17/2022.
//

#ifndef SEANN_2_CUCONV_CUH
#define SEANN_2_CUCONV_CUH

#include "../tensor/Tensor.cuh"
#include "../assist/TensorAssert.cuh"
#include "../assist/Inspection.cuh"

namespace seblas {
    /**
     * Convolution operator by GEMM (will implement FFT and vinograd in future)
     * @param A filters OC * IC * FH * FW
     * @param B feature inputs N * IC * IH * IW
     * @param C feature outputs N ＊ OC * OH * OW
     * @param biases set this to be nullpointer if you don't want biases (OC * 1)
     * @return
     */
    Tensor *conv(Tensor *A, Tensor *B, Tensor *C, int strideH, int strideW, int padH, int padW, Tensor *biases);

    /**
     * The back propagation of conv layers with respect to input features
     * @param A filters OC * IC * FH * FW
     * @param B feature outputs N * OC * OH * OW
     * @param C feature inputs N * IC * IH * IW
     * @return
     */
    Tensor *convDerive(Tensor *A, Tensor *B, Tensor *C, int strideH, int strideW, int padH, int padW);

    /**
     * Calculate gradients of filters (weights) based on errors of the output features
     * This method will loop over all elements on the "ON" dimension and add the errors up
     * the final deltas will be the sum of errors on ON dimension divide by ON
     * @param A errors in : ON * OC * OH * OW
     * @param B input features : ON * IC * IH * IW
     * @param C filters : OC * IC * FH * FW
     * @return
     */
     //TODO: implement this and add BN support
    Tensor* convError(Tensor *A, Tensor *B, Tensor *C, int strideH, int strideW, int padH, int padW);
}

#endif //SEANN_2_CUCONV_CUH
