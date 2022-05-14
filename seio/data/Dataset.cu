//
// Created by Dylan on 5/14/2022.
//

#include "Dataset.cuh"

namespace seio{

    Data *Data::declare(shape4 dataShape, shape4 labelShape) {
        Data* out;
        cudaMallocHost(&out, sizeof(Data));
        out->X = Tensor::declare(dataShape);
        out->label = Tensor::declare(labelShape);
        return out;
    }

    Data *Data::create() {
        X->create();
        label->create();
        return this;
    }

    Data *Data::createHost() {
        X->createHost();
        label->createHost();
        return this;
    }

    Data *Data::inherit(Tensor *X0, Tensor *label0) {
        X->attach(X0);
        label->attach(label0);
        return this;
    }

    Data *Data::copyOffD2D(Data *onDevice) {
        onDevice->X->copyToD2D(X);
        onDevice->label->copyToD2D(label);
        return this;
    }

    Data *Data::copyOffH2D(Data *onHost) {
        onHost->X->copyToH2D(X);
        onHost->label->copyToH2D(label);
        return this;
    }

    Data *Data::copyOffD2H(Data *onDevice) {
        onDevice->X->copyToD2H(X);
        onDevice->label->copyToD2H(label);
        return this;
    }

    Data *Data::copyOffH2H(Data *onHost) {
        onHost->X->copyToH2H(X);
        onHost->label->copyToH2H(label);
        return this;
    }

    Data *Data::copyOffD2D(Tensor *X0, Tensor *label0) {
        X0->copyToD2D(X);
        label0->copyToD2D(label);
        return this;
    }

    Data *Data::copyOffH2D(Tensor *X0, Tensor *label0) {
        X0->copyToH2D(X);
        label0->copyToH2D(label);
        return this;
    }

    Data *Data::copyOffD2H(Tensor *X0, Tensor *label0) {
        X0->copyToD2H(X);
        label0->copyToD2H(label);
        return this;
    }

    Data *Data::copyOffH2H(Tensor *X0, Tensor *label0) {
        X0->copyToH2H(X);
        label0->copyToH2H(label);
        return this;
    }

    void Data::destroy() {
        X->destroy();
        label->destroy();
        cudaFree(this);
    }

    void Data::destroyHost() {
        X->destroyHost();
        label->destroyHost();
        cudaFreeHost(this);
    }

    Dataset* Dataset::construct(uint32 batchSize, uint32 epochSize,
                              shape4 dataShape, shape4 labelShape) {
        Dataset* out;
        cudaMallocHost(&out, sizeof(Dataset));
        out->BATCH_SIZE = batchSize;
        out->EPOCH_SIZE = epochSize;
        cudaMallocHost(&out->dataBatch[0], batchSize * sizeof(Data*));
        cudaMallocHost(&out->dataBatch[1], batchSize * sizeof(Data*));

        for(uint32 i = 0; i < batchSize; i++) {
            out->dataBatch[0][i] = Data::declare(dataShape, labelShape)->create();
            out->dataBatch[1][i] = Data::declare(dataShape, labelShape)->create();
        }
        return out;
    }

    void Dataset::genBatch() {
        auto p1 = std::chrono::system_clock::now();
        default_random_engine generator(
                chrono::duration_cast<std::chrono::seconds>(
                        p1.time_since_epoch()).count()
                );

        for(int i = 0; i < BATCH_SIZE; i++) {

            //find the current epoch parameters
            uint32 batchOperateID = batchID % 2;
            uint32 epochOperateID = epochID % 2;
            uniform_int_distribution<uint32> distribution(0, dataset[epochOperateID].size() - 1);
            uint32 index = distribution(generator);

            //copy data to CUDA memory
            dataBatch[batchOperateID][i]->copyOffH2D(dataset[epochOperateID][index]);

            //change data location to prevent repeating use of same data
            dataset[epochOperateID^1].push_back(dataset[epochOperateID][index]);
            dataset[epochOperateID].erase(dataset[epochOperateID].begin() + index);

            if(dataset[epochOperateID].empty()) {
                epochID++;
            }

            //change batchID
            batchID++;
        }
    }

    void genBatchThread(uint32 BATCH_SIZE, uint32 EPOCH_SIZE,
                        uint32* bID, uint32* eID, vector<Data*>* dataset,
                        Data*** dataBatch) {
        auto p1 = std::chrono::system_clock::now();
        uint32 batchID = *bID;
        uint32 epochID = *eID;
        default_random_engine generator(
                chrono::duration_cast<std::chrono::seconds>(
                        p1.time_since_epoch()).count()
        );

        for(int i = 0; i < BATCH_SIZE; i++) {

            //find the current epoch parameters
            uint32 batchOperateID = batchID % 2;
            uint32 epochOperateID = epochID % 2;
            uniform_int_distribution<uint32> distribution(0, dataset[epochOperateID].size() - 1);
            uint32 index = distribution(generator);

            //copy data to CUDA memory
            dataBatch[batchOperateID][i]->copyOffH2D(dataset[epochOperateID][index]);

            //change data location to prevent repeating use of same data
            dataset[epochOperateID^1].push_back(dataset[epochOperateID][index]);
            dataset[epochOperateID].erase(dataset[epochOperateID].begin() + index);

            if(dataset[epochOperateID].empty()) {
                epochID++;
            }

            //change batchID
            batchID++;
        }
        *bID = batchID;
        *eID = epochID;
    }

    //this method uses an async way of generating the next batch data
    //while training is running on the current batch
    thread Dataset::genBatchAsync() {
        thread t1 = thread(genBatchThread, BATCH_SIZE, EPOCH_SIZE,
                           &batchID, &epochID, dataset, dataBatch);
        return t1;
    }
}