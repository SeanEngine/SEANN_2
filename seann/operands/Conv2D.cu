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
        convError(Y->grad, X->a, filter->A->grad, (int)strideH, (int)strideW, (int)padH, (int)padW);
        if(WITH_BIAS){
            bias->A->grad = channelReduce(Y->grad, bias->A->grad, reduceBuf);
        }
    }

    void Conv2D::updateParams() {
        filter->opt->apply();
        if(WITH_BIAS){
            bias->opt->apply();
        }
    }

    void Conv2D::batchUpdateParams() {
        filter->opt->batchApply();
        if(WITH_BIAS){
            bias->opt->batchApply();
        }
    }
}