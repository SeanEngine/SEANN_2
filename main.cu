#include <iostream>
#include "seblas/operations/cuOperations.cuh"
#include "seann/seann.cuh"


using namespace seblas;
using namespace seann;

int main(int argc, char** argv) {
    auto* model = new Sequential({
        new Conv2D(shape4(3,32,32), shape4(3,3,3,3),1,1,1,1,false),
        new ReLU(3072),
        new Conv2D(shape4(3,32,32), shape4(3,3,3,3),1,1,1,1,false),
        new ReLU(3072),
        new Conv2D(shape4(3,32,32), shape4(3,3,3,3),2,2,1,1,false),
        new ReLU(768),
        new Conv2D(shape4(3,16,16), shape4(3,3,3,3),1,1,1,1,false),
        new ReLU(768),
        new Linear(768,768),
        new ReLU(768),
        new Linear(768,10),
        new Softmax(10)
    });

    OptimizerInfo* info = new OPTIMIZER_ADAM(0.001, 0.9, 0.999, 1e-8);

    model->construct(info);
    model->waive();
    model->randInit();
    model->setLoss(CrossEntropyLoss);

    auto* data = Tensor::declare(3,32,32)->create()->randNormal(2,3);
    inspect(model->forward(data));
    auto* label = Tensor::declare(10,1)->create()->randNormal(0,0.3);
    inspect(model->backward(label));
    model->learn();
}