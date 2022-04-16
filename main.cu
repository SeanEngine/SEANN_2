#include <iostream>
#include "seblas/tensor/Tensor.cuh"
#include <vector>

using namespace seblas;

int main() {
    Tensor* t = Tensor::declare(10,15)->create()->randNormal(2,0.5)->toHost();
    for(int i=0; i < t->dims.size; i++){
        std::cout << t->elements[i] << " ";
    }
}
