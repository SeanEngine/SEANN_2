#include <iostream>
#include "seblas/operations/cuOperations.cuh"
#include "seann/optimizers/Optimizer.cuh"

using namespace seblas;
using namespace seann;

int main(int argc, char** argv) {
    auto* param = Parameter::create(10,29);
    param->grad->constFill(0.1);
    param->a->constFill(0);
    inspect(param->a);

    Adam* optim = new Adam(0.6,param);
    optim->apply();
    inspect(param->a);
}