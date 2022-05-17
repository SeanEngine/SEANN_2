//
// Created by Dylan on 5/13/2022.
//

#ifndef SEANN_2_LOSS_CUH
#define SEANN_2_LOSS_CUH
#include "../components/Parameter.cuh"
#include "../../seblas/operations/cuOperations.cuh"

namespace seann {
    typedef void(*LossFunc)(Parameter*, Tensor*);
    typedef float(*LossFuncCalc)(Parameter*, Tensor*, Tensor*);

    void crossEntropyLoss(Parameter* Y, Tensor* label);

    float crossEntropyCalc(Parameter* Y, Tensor* label, Tensor* buf);


} // seann

#endif //SEANN_2_LOSS_CUH
