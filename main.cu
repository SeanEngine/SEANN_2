#include <iostream>
#include "seblas/operations/cuOperations.cuh"

using namespace seblas;

int main(int argc, char** argv) {
    auto* A = Tensor::declare(10,1025)->create()->randNormal(3,0.1);
    auto* B = Tensor::declare(10,1025)->create();
    auto* buf = Tensor::declare(10,2)->create();

    softmax(A,B,buf, 1025);
    inspect(B);

    auto* C = Tensor::declare(10,1)->create();

    rowReduce(B,C,buf);
    inspect(C);
}