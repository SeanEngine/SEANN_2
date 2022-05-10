//
// Created by Dylan on 5/8/2022.
//

#ifndef SEANN_2_CONV2D_CUH
#define SEANN_2_CONV2D_CUH

#include "OperandBase.cuh"
#include "../../seblas/assist/TensorAssert.cuh"

namespace seann {
    class Conv2D : public OperandBase {
    public:
        shape4 filterShape;
        NetParam* filter{};
        NetParam* bias = nullptr;
        uint32 strideH;
        uint32 strideW;
        uint32 padH;
        uint32 padW;
        bool WITH_BIAS = false;

        Conv2D(shape4 inShape, shape4 filterShape, uint32 strideH, uint32 strideW, uint32 padH, uint32 padW, bool WITH_BIAS)
        : filterShape(filterShape), strideH(strideH), strideW(strideW), padH(padH), padW(padW) {
            X = Parameter::declare(inShape); //input features
            shape4 outShape = {
                    filterShape.n,
                    (inShape.h + 2 * padH - filterShape.h) / strideH + 1,
                    (inShape.w + 2 * padW - filterShape.w) / strideW + 1};

            Y = Parameter::create(outShape);
            this->WITH_BIAS = WITH_BIAS;
        }

        void initNetParams(OptimizerInfo *info) override {
            filter = new NetParam(info, filterShape);
            if (WITH_BIAS) bias = new NetParam(info, filterShape.n, 1);
        }

        void forward() override;

        void xGrads() override;

        void paramGrads() override;
    };
}

#endif //SEANN_2_CONV2D_CUH
