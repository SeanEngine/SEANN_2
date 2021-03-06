//
// Created by Dylan on 5/8/2022.
//

#include "Conv2D.cuh"

namespace seann {
    void Conv2D::forward() {
        if(WITH_BIAS){
            conv(filter->data(), X->a, Y->a,
                 (int)strideH, (int)strideW, (int)padH, (int)padW, bias->data());
            return;
        }
        conv(filter->data(), X->a, Y->a,
             (int)strideH, (int)strideW, (int)padH, (int)padW, nullptr);
    }

    void Conv2D::xGrads() {
        convDerive(filter->data(), Y->grad, X->grad, (int)strideH, (int)strideW, (int)padH, (int)padW);
    }

    void Conv2D::paramGrads() {
        convError(Y->grad, X->a, filter->grad(), (int)strideH, (int)strideW, (int)padH, (int)padW);
        if(WITH_BIAS) {
            channelReduce(Y->grad, bias->grad(), reduceBuf);
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

    void Conv2D::randFillNetParams() {
        uint32 K = filter->data()->dims.size / filter->data()->dims.n;
        filter->data()->randNormal(0, (float)sqrt(2.0 / (float) K));
        if (WITH_BIAS)
            bias->data()->randNormal(0, (float)sqrt(2.0 / (float)filter->data()->dims.n));
    }
}