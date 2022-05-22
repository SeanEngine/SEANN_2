#include <iostream>
#include "seblas/operations/cuOperations.cuh"
#include "seann/seann.cuh"
#include "seio/data/DataLoader.cuh"


using namespace seblas;
using namespace seann;

int main(int argc, char** argv) {

    auto* model = new Sequential({
        new Conv2D(shape4(1,28,28), shape4(6,1,5,5), 1,1,0,0, false),
        new ReLU(shape4(6,24,24).size),
        new MaxPool2D(shape4(6,24,24),2,2),

        new Conv2D(shape4(6,12,12),shape4(16,6,5,5),1,1,0,0, false),
        new ReLU(shape4(16,8,8).size),
        new MaxPool2D(shape4(16,8,8),2,2),

        new Linear(256,120),
        new ReLU(120),
        new Linear(120,10),
        new Softmax(10)
    });

    OptimizerInfo* info = new OPTIMIZER_SGD(0.003);

    model->construct(info);
    model->waive();
    model->randInit();
    model->setLoss(crossEntropyLoss, crossEntropyCalc);

    auto* dataset = Dataset::construct(100,60000,100,shape4(28,28),shape4(10,1));
    fetchIDX(dataset, R"(D:\Resources\Datasets\mnist-bin\train-images.idx3-ubyte)",784, false);
    fetchIDX(dataset, R"(D:\Resources\Datasets\mnist-bin\train-labels.idx1-ubyte)",1, true);

    model->train(dataset);
}