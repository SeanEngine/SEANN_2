//
// Created by Dylan on 5/11/2022.
//

#ifndef SEANN_2_SEQUENTIAL_CUH
#define SEANN_2_SEQUENTIAL_CUH

#include "../components/Parameter.cuh"
#include "../operands/OperandBase.cuh"

namespace seann {
    class Sequential {
    public:
        Parameter* netX;
        Parameter* netY;

        //a list of pointers to all the initialized operands
        OperandBase** operands{};
        uint32 OPERAND_COUNT;

        template <class T>
        Sequential(std::initializer_list<T*> list){
            OPERAND_COUNT = list.size();
            cudaMallocHost(&operands, OPERAND_COUNT * sizeof(OperandBase*));
            for(auto i = 0; i < OPERAND_COUNT; i++) {
                operands[i] = list[i];
            }

            //bind inputs and outputs
            netX = Parameter::create(operands[0]->X->a->dims);
            operands[0]->X->inherit(netX);
            netY = Parameter::declare(operands[OPERAND_COUNT-1]->Y->a->dims)
                    ->inherit(operands[OPERAND_COUNT-1]->Y);

            logInfo(seio::LOG_SEG_SEANN, "Constructing Model: ");
            for(auto i = 0; i < OPERAND_COUNT; i++) {
                logInfo(seio::LOG_SEG_SEANN, operands[i]->info(), seio::LOG_COLOR_LIGHT_BLUE);
            }
        }

        void waive() const;

        void construct(OptimizerInfo* info) const;
    };
} // seann

#endif //SEANN_2_SEQUENTIAL_CUH
