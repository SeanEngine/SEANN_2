//
// Created by Dylan on 5/14/2022.
//

#ifndef SEANN_2_DATASET_CUH
#define SEANN_2_DATASET_CUH

#include "../../seblas/tensor/Tensor.cuh"
#include <random>
#include <chrono>
#include <thread>

using namespace seblas;
using namespace std;

namespace seio {
    struct Data{
    public:
        Tensor* X;
        Tensor* label;

        static Data* declare(shape4 dataShape, shape4 labelShape);

        Data* create();

        Data* createHost();

        void destroy();

        void destroyHost();

        Data* inherit(Tensor* X0, Tensor* label0);

        Data* copyOffD2D(Data* onDevice);

        Data* copyOffH2D(Data* onHost);

        Data* copyOffD2H(Data* onDevice);

        Data* copyOffH2H(Data* onHost);

        Data* copyOffD2D(Tensor* X0, Tensor* label0);

        Data* copyOffH2D(Tensor* X0, Tensor* label0);

        Data* copyOffD2H(Tensor* X0, Tensor* label0);

        Data* copyOffH2H(Tensor* X0, Tensor* label0);
    };

    struct Dataset {
    public:
        //onDevice
        Data** dataBatch[2];

        //onHost : supports epoch operations
        vector<Data*> dataset[2];

        uint32 BATCH_SIZE;
        uint32 EPOCH_SIZE;

        uint32 batchID = 0;
        uint32 epochID = 0;

        static Dataset* construct(uint32 batchSize, uint32 epochSize,
                                  shape4 dataShape, shape4 labelShape);

        //generate a data batch
        void genBatch();

        //this method uses an async way of generating the next batch data
        //while training is running on the current batch
        thread genBatchAsync();


    };
}


#endif //SEANN_2_DATASET_CUH
