#include <iostream>
#include "seblas/operations/cuOperations.cuh"
#include "seann/optimizers/Optimizer.cuh"

using namespace seblas;
using namespace seann;

int main(int argc, char** argv) {
    Tensor* B = Tensor::declare(2, 3, 32, 32)->create()->randNormal(1,0);
    Tensor* A = Tensor::declare(2, 6, 16, 16)->create()->randNormal(1,0);
    Tensor* C = Tensor::declare(6,3,3,3)->create();

    convError(A,B,C,2,2,1,1);

    inspect(C);
}