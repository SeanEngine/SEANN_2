#include <iostream>
#include "seblas/operations/cuOperations.cuh"

using namespace seblas;

int main(int argc, char** argv) {
    auto* A = Tensor::declare(10,1023)->create()->constFill(3);
    auto* B = Tensor::declare(10,1)->create();
    auto* buf = Tensor::declare(10,1)->create();

    reduce(A,B,buf,1023);
    inspect(B);
}