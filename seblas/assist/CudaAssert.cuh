//
// Created by DanielSun on 4/16/2022.
//

#ifndef SEANN_2_CUDAASSERT_CUH
#define SEANN_2_CUDAASSERT_CUH

#include "../../seio/logging/LogUtils.cuh"

using namespace seio;
namespace seblas{
    void assertCuda(const char* file, int line);
}


#endif //SEANN_2_CUDAASSERT_CUH
