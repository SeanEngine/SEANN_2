#include <iostream>
#include "seblas/tensor/Tensor.cuh"
#include "seblas/operations/cuGEMM.cuh"
#include <vector>

using namespace seblas;

int main() {
    auto* A = Tensor::declare(80,40)->create()->randNormal(2,3);
    auto* B = Tensor::declare(40,20)->create()->randNormal(2,3);
    auto* C = Tensor::declare(80,20)->create();

    sgemm(A,B,C);
}
