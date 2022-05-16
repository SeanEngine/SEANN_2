//
// Created by Dylan on 5/11/2022.
//

#ifndef SEANN_2_SEQUENTIAL_CUH
#define SEANN_2_SEQUENTIAL_CUH

#include "../components/Parameter.cuh"
#include "../operands/OperandBase.cuh"
#include "../optimizers/Loss.cuh"
#include "../../seio/data/Dataset.cuh"

using namespace seio;

namespace seann {
    class Sequential {
    public:
        Parameter* netX;
        Parameter* netY;
        LossFunc loss{};

        //a list of pointers to all the initialized operands
        OperandBase** operands{};
        uint32 OPERAND_COUNT;

        Sequential(std::initializer_list<OperandBase*> list){
            OPERAND_COUNT = list.size();
            cudaMallocHost(&operands, OPERAND_COUNT * sizeof(OperandBase*));
            for(auto i = 0; i < OPERAND_COUNT; i++) {
                operands[i] = list.begin()[i];
            }

            //bind inputs and outputs
            netX = Parameter::create(operands[0]->X->a->dims);
            operands[0]->X->inherit(netX);
            netY = Parameter::declare(operands[OPERAND_COUNT-1]->Y->a->dims)
                    ->inherit(operands[OPERAND_COUNT-1]->Y);
        }

        void waive() const;

        void construct(OptimizerInfo* info) const;

        void setLoss(LossFunc loss);

        void randInit() const;

        Tensor* forward() const;

        Tensor* forward(Tensor* X) const;

        Tensor* backward(Tensor* labelY) const;

        void learn() const;

        void learnBatch() const;

        void train(Dataset* data) const;
    };
} // seann

#endif //SEANN_2_SEQUENTIAL_CUH
