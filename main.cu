#include <iostream>
#include "seblas/operations/cuOperations.cuh"
#include "seann/seann.cuh"
#include "seio/data/DataLoader.cuh"


using namespace seblas;
using namespace seann;

int main(int argc, char** argv) {

    auto* model = new Sequential({
        new Linear(784, 120),
        new ReLU(120),
        new Linear(120,32),
        new ReLU(32),
        new Linear(32,10),
        new Softmax(10)
    });

    OptimizerInfo* info = new OPTIMIZER_ADADELTA(0.01);

    model->construct(info);
    model->waive();
    model->randInit();
    model->setLoss(crossEntropyLoss, crossEntropyCalc);

    auto* dataset = Dataset::construct(100,60000,200,shape4(28,28),shape4(10,1));
    fetchIDX(dataset, R"(D:\Resources\Datasets\mnist-bin\train-images.idx3-ubyte)",784, false);
    fetchIDX(dataset, R"(D:\Resources\Datasets\mnist-bin\train-labels.idx1-ubyte)",1, true);

    model->train(dataset);
}