//
// Created by Dylan on 5/8/2022.
//

#include "Conv2D.cuh"

namespace seann {
    void Conv2D::forward() {
        if(WITH_BIAS){
            conv(filter->A->a, X->a, Y->a,
                 (int)strideH, (int)strideW, (int)padH, (int)padW, bias->A->a);
            return;
        }
        conv(filter->A->a, X->a, Y->a,
             (int)strideH, (int)strideW, (int)padH, (int)padW, nullptr);
    }

    void Conv2D::xGrads() {
        convDerive(filter->A->a, Y->grad, X->grad, (int)strideH, (int)strideW, (int)padH, (int)padW);
    }

    void Conv2D::paramGrads() {

    }
}