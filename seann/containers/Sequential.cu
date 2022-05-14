//
// Created by Dylan on 5/11/2022.
//

#include "Sequential.cuh"

namespace seann {
    void Sequential::waive() const {
        for (int i = 1; i < OPERAND_COUNT; i++) {
            operands[i]->bindPrev(operands[i - 1]);
        }

        for(int i = 0; i < OPERAND_COUNT-1; i++) {
            operands[i]->bindNext(operands[i + 1]);
        }
    }

    void Sequential::construct(OptimizerInfo* info) const {
        logInfo(seio::LOG_SEG_SEANN, "Constructing Model: ");
        for (int i = 0; i < OPERAND_COUNT; i++) {
            operands[i]->initNetParams(info);
            logDebug(seio::LOG_SEG_SEANN, operands[i]->info());
        }
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

    void Sequential::setLoss(LossFunc lossFunc) {
        loss = lossFunc;
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
} // seann