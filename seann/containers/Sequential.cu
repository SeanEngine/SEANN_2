//
// Created by Dylan on 5/11/2022.
//

#include "Sequential.cuh"
#include "../operands/Softmax.cuh"

namespace seann {
    void Sequential::waive() const {
        for (int i = 1; i < OPERAND_COUNT; i++) {
            operands[i]->bindPrev(operands[i - 1]);
        }

        for(int i = 0; i < OPERAND_COUNT-1; i++) {
            operands[i]->bindNext(operands[i + 1]);
        }
    }

    void Sequential::construct(OptimizerInfo* info) {
        logInfo(seio::LOG_SEG_SEANN, "Constructing Model: ");
        for (int i = 0; i < OPERAND_COUNT; i++) {
            operands[i]->initNetParams(info, i == 0 ? netX->a->dims : operands[i-1]->Y->a->dims);
            logInfo(seio::LOG_SEG_SEANN, operands[i]->info());
        }

        //bind inputs and outputs
        operands[0]->X->inherit(netX);
        netY = Parameter::declare(operands[OPERAND_COUNT-1]->Y->a->dims)
                ->inherit(operands[OPERAND_COUNT-1]->Y);

        //bind the layers together
        waive();
    }

    void fillData(Parameter* netX, Data** data, uint32 beg){
        for(int i = 0; i < netX->a->dims.n; i++){
            cudaMemcpy(netX->a->elements + (netX->a->dims.size / netX->a->dims.n) * i,
                       data[beg + i]->X->elements,
                       netX->a->dims.size / netX->a->dims.n * sizeof(float),
                       cudaMemcpyDeviceToDevice);
        }
        assertCuda(__FILE__, __LINE__);
    }

    void fillLabel(Tensor* labels, Data** data, uint32 beg){
        for(int i = 0; i < labels->dims.n; i++){
            cudaMemcpy(labels->elements + (labels->dims.size / labels->dims.n) * i,
                       data[beg + i]->label->elements,
                       labels->dims.size / labels->dims.n * sizeof(float),
                       cudaMemcpyDeviceToDevice);
        }
        assertCuda(__FILE__, __LINE__);
    }

    Tensor* Sequential::forward() const {
        for (int i = 0; i < OPERAND_COUNT; i++) {
            operands[i]->forward();
        }
        return netY->a;
    }

    Tensor* Sequential::forward(Tensor* X) const {
        X->copyToD2D(netX->a);
        for (int i = 0; i < OPERAND_COUNT; i++) {
            operands[i]->forward();
        }
        return netY->a;
    }

    void Sequential::randInit() const {
        for (int i = 0; i < OPERAND_COUNT; i++) {
            operands[i]->randFillNetParams();
        }
    }

    Tensor *Sequential::backward(Tensor* label) const {
        loss(netY, label);
        for (int i = (int)OPERAND_COUNT - 1; i >= 0; i--) {
            operands[i]->xGrads();
            operands[i]->paramGrads();
        }
        return netX->grad;
    }

    void Sequential::setLoss(LossFunc lossFunc, LossFuncCalc lossFWD) {
        loss = lossFunc;
        lossFW = lossFWD;
    }

    void Sequential::learn() const {
        for (int i = 0; i < OPERAND_COUNT; i++) {
            operands[i]->updateParams();
        }
    }

    void Sequential::learnBatch() const {
        for (int i = 0; i < OPERAND_COUNT; i++) {
            operands[i]->batchUpdateParams();
        }
    }

    //this train method does not support BN
    void Sequential::train(Dataset *data) const {
        assert(data->BATCH_SIZE > 0 && data->BATCH_SIZE % netX->a->dims.n == 0);
        data->genBatch();
        auto* labels = Tensor::declare(netY->a->dims)->create();
        while(data->epochID < data->MAX_EPOCH){
            uint32 batchID = data->batchID-1;
            auto pass = data->genBatchAsync();
            float batchLoss = 0;

            //training over each sample in the batch
            for(uint32 sampleID = 0; sampleID < data->BATCH_SIZE; sampleID+= netX->a->dims.n){
                fillData(netX, data->dataBatch[batchID % 2], sampleID);
                forward();
                fillLabel(labels, data->dataBatch[batchID % 2], sampleID);
                backward(labels);

                float lossVal = lossFW(netY, labels);
                batchLoss += lossVal;
                learn();


            }

            if(data->batchID % 5 == 0) {
                //inspect(operands[OPERAND_COUNT-2]->Y->a);
                cout << batchLoss / (float) data->BATCH_SIZE << ", ";
            }

            //BGD updates
            learnBatch();
            pass.join(); //wait for next batch to prefetch
        }
    }
} // seann