#include <iostream>
#include "seblas/tensor/Tensor.cuh"
#include <vector>

using namespace seblas;

int main() {
    Tensor* t = Tensor::declare(3,15)->createHost()->toDevice();
}
