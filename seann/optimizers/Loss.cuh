//
// Created by Dylan on 5/13/2022.
//

#ifndef SEANN_2_LOSS_CUH
#define SEANN_2_LOSS_CUH
#include "../components/Parameter.cuh"

namespace seann {
    typedef void(*LossFunc)(Parameter*, Tensor*);

    void CrossEntropyLoss(Parameter* Y, Tensor* label);
} // seann

#endif //SEANN_2_LOSS_CUH
