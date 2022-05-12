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
        Tensor* reduceBuf; //for calculating gradients of bias

        Conv2D(shape4 inShape, shape4 filterShape, uint32 strideH, uint32 strideW, uint32 padH, uint32 padW, bool WITH_BIAS)
        : filterShape(filterShape), strideH(strideH), strideW(strideW), padH(padH), padW(padW) {
            X = Parameter::declare(inShape); //input features
            shape4 outShape = {
                    filterShape.n,
                    (inShape.h + 2 * padH - filterShape.h) / strideH + 1,
                    (inShape.w + 2 * padW - filterShape.w) / strideW + 1};

            Y = Parameter::create(outShape);
            this->WITH_BIAS = WITH_BIAS;
            if(WITH_BIAS) {
                reduceBuf = outShape.h * outShape.w > 1024 ?
                        Tensor::declare(filterShape.n, outShape.h * outShape.w / 1024) : nullptr;
            }
        }

        string info() override {
            return "Conv2D { filter: " + filter->A->a->dims.toString() + ", input feature: " + Y->a->dims.toString() + " }";
        }

        void initNetParams(OptimizerInfo *info) override {
            filter = new NetParam(info, filterShape);
            if (WITH_BIAS) bias = new NetParam(info, filterShape.n, 1);
        }

        void forward() override;

        void xGrads() override;

        void paramGrads() override;

        void updateParams() override;

        void batchUpdateParams() override;
    };
}

#endif //SEANN_2_CONV2D_CUH
