//
// Created by DanielSun on 4/16/2022.
//

#include "CudaAssert.cuh"

using namespace std;
namespace seblas{
    void assertCuda(const char* file, int line){
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess){
            logFatal(seio::LOG_SEG_SEBLAS, string("Device error encountered:") + cudaGetErrorString(error));
            logFatal(seio::LOG_SEG_SEBLAS, "line: " + to_string(line) + "  file: " + string(file));
            throw runtime_error("line: " + to_string(line) + "  file: " + string(file));
        }
    }
}