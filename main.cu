#include <iostream>
#include "seblas/operations/cuOperations.cuh"
#include "seann/seann.cuh"
#include "seio/data/DataLoader.cuh"


using namespace seblas;
using namespace seann;

int main(int argc, char** argv) {

    auto* model = new Sequential({

        new Conv2D(shape4(1,28,28), shape4(1,1,3,3),1,1,1,1,false),
        new ReLU(784),
        new Conv2D(shape4(1,28,28), shape4(1,1,3,3),1,1,1,1,false),
        new ReLU(784),
        new Conv2D(shape4(1,28,28), shape4(1,1,3,3),1,1,1,1,false),
        new ReLU(784),
        new Conv2D(shape4(1,28,28), shape4(1,1,3,3),1,1,1,1,false),
        new ReLU(784),

        new Linear(784, 784),
        new ReLU(784),
        new Linear(784,120),
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

/*
    auto* X = Parameter::create(2,5,5);
    auto* filter = Parameter::create(6,2,3,3);
    auto* Y = Parameter::create(6,5,5);

    X->a->randNormal(1,1);
    inspect(X->a);
    cout<<"========"<<endl;
    filter->a->randNormal(1,1);
    inspect(filter->a);
    cout<<"========"<<endl;
    Y->grad->randNormal(1,2);
    inspect(Y->grad);
    cout<<"========"<<endl;

    conv(filter->a, X->a, Y->a, 1,1,1,1, nullptr);
    inspect(Y->a);
    cout<<"######"<<endl;

    convDerive(filter->a, Y->grad, X->grad, 1,1,1,1);
    inspect(X->grad);
    cout<<"######"<<endl;

    convError(Y->grad, X->a, filter->grad,1,1,1,1);
    inspect(filter->grad);
    */
}