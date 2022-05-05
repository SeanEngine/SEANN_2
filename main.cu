#include <iostream>
#include "seblas/operations/cuOperations.cuh"
#include "seann/optimizers/Optimizer.cuh"

using namespace seblas;
using namespace seann;

int main(int argc, char** argv) {
    auto* param = Parameter::create(20,30);
    param->grad->randNormal(0,2);
    param->a->constFill(1);
    param->grad->copyD2D(param->a);
    inspect(param->a);
}