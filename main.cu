#include <iostream>
#include "seblas/operations/cuOperations.cuh"
#include "seann/seann.cuh"
#include "seio/data/DataLoader.cuh"


using namespace seblas;
using namespace seann;

int main(int argc, char** argv) {

    auto* model = new Sequential({
        new Conv2D(shape4(1,28,28), shape4(3,1,3,3),1,1,1,1,false),
        new ReLU(2352),
        new Conv2D(shape4(3,28,28), shape4(3,3,3,3),1,1,1,1,false),
        new ReLU(2352),
        new Conv2D(shape4(3,28,28), shape4(6,3,3,3),2,2,1,1,false),
        new ReLU(1176),
        new Conv2D(shape4(6,14,14), shape4(6,6,3,3),1,1,1,1,false),
        new ReLU(1176),
        new Linear(1176,768),
        new ReLU(768),
        new Linear(768,10),
        new Softmax(10)
    });

    OptimizerInfo* info = new OPTIMIZER_ADAM(0.001, 0.9, 0.999, 1e-8);

    model->construct(info);
    model->waive();
    model->randInit();
    model->setLoss(CrossEntropyLoss);

    auto* dataset = Dataset::construct(100,60000,200,shape4(28,28),shape4(10,1));
    fetchIDX(dataset, R"(D:\Resources\Datasets\mnist-bin\train-images.idx3-ubyte)",784, false);
    fetchIDX(dataset, R"(D:\Resources\Datasets\mnist-bin\train-labels.idx1-ubyte)",1, true);


}