#include <iostream>
#include "seblas/operations/cuOperations.cuh"
#include "seann/seann.cuh"
#include "seio/data/DataLoader.cuh"


using namespace seblas;
using namespace seann;

int main(int argc, char** argv) {


    auto* model = new Sequential({
        /*
         new Conv2D(shape4(3,32,32), shape4(32,3,3,3), 1,1,1,1, false),
         new ReLU(shape4(32,32,32).size),
         new Conv2D(shape4(32,32,32), shape4(32,32,3,3), 1,1,1,1, false),
         new ReLU(shape4(32,32,32).size),
         new MaxPool2D(shape4(32,32,32),2,2),

         new Conv2D(shape4(32,16,16), shape4(64,32,3,3), 1,1,1,1, false),
         new ReLU(shape4(64,16,16).size),
         new Conv2D(shape4(64,16,16), shape4(64,64,3,3), 1,1,1,1, false),
         new ReLU(shape4(64,16,16).size),
         new MaxPool2D(shape4(64,16,16),2,2),

         new Conv2D(shape4(64,8,8), shape4(128,64,3,3), 1,1,1,1, false),
         new ReLU(shape4(128,8,8).size),
         new Conv2D(shape4(128,8,8), shape4(128,128,3,3), 1,1,1,1, false),
         new ReLU(shape4(128,8,8).size),
         new MaxPool2D(shape4(128,8,8),2,2),
         */

         new Linear(3072,120),
         new ReLU(120),
         new Linear(120,32),
         new ReLU(32),
         new Linear(32,10),
         new Softmax(10)
    });

    OptimizerInfo* info = new OPTIMIZER_SGD(0.003);

    model->construct(info);
    model->waive();
    model->randInit();
    model->setLoss(crossEntropyLoss, crossEntropyCalc);

    auto* dataset = Dataset::construct(100,50000,100,shape4(3,32,32),shape4(10,1));
    const char * BASE_PATH = R"(D:\Resources\Datasets\cifar-10-bin\data_batch_)";
    for(int i = 0; i < 5; i++){
        string binPath = BASE_PATH + to_string(i+1) + ".bin";
        fetchCIFAR(dataset, binPath.c_str(), i);
    }

    //inspect(dataset->dataset[5000]->X);
    //inspect(dataset->dataset[5000]->label);

    model->train(dataset);

}