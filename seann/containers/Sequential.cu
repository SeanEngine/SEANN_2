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
} // seann