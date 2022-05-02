//
// Created by Dylan on 5/1/2022.
//

#ifndef SEANN_2_OPTIMIZER_CUH
#define SEANN_2_OPTIMIZER_CUH

#include "../components/Parameter.cuh"
#include "../../seblas/operations/cuOperations.cuh"

using namespace seblas;

namespace seann {
    /**
     * @brief for all the optimisers, exists the following functions:
     *
     * parameter to optimize : θ, cost function : C(θ), LEARNING_RATE : η
     *
     * 1. calculate the gradients of parameters to update: gt = ▽C(θt)
     * 2. calculate 1st and 2nd order momentum: m = Φ(g1,g2,...,gt), v = φ(g1,g2,...,gt)
     * 3. calculate update amount: θt' = η * m / (sqrt(v) + ε)
     * 4. update parameters: θt+1 = θt - θt'
     */
    struct Optimizer {
    public:
        float LEARNING_RATE{};
        Parameter* target;

        explicit Optimizer(float LEARNING_RATE, Parameter* target)
           : LEARNING_RATE(LEARNING_RATE), target(target){}


        //apply the gradient to the parameters (weights, biases, etc)
        virtual void apply() = 0;

        //updates relative to batches
        virtual void batchApply() = 0;
    };


    //SGD optimizer : Stochastic Gradient Decent, w = w - η * g
    //This optimizer does not have batch behaviors
    struct SGD : public Optimizer {
    public:
        explicit SGD(float LEARNING_RATE, Parameter* target)
            : Optimizer(LEARNING_RATE, target){}


        void apply() override {
            target->a = *target->a - (*target->grad * LEARNING_RATE);
            target->grad->constFill(0);
        }

        void batchApply() override{}
    };


    //BGD optimizer : Batch Gradient Decent, w = w - (η * sum_batch(g))/bs
    struct BGD : public Optimizer {
    public:
        float BATCH_SIZE;
        explicit BGD(float LEARNING_RATE, Parameter* target, float BATCH_SIZE)
            : Optimizer(LEARNING_RATE, target), BATCH_SIZE(BATCH_SIZE){}

        void apply() override {}

        void batchApply() override{
            target->a = *target->a - (*target->grad * (LEARNING_RATE/BATCH_SIZE));
            target->grad->constFill(0);
        }
    };


    //Momentum : m[t] = m[t-1] * β + (1 - β) * g[t]
    //           w[t] = w[t-1] - η * m[t]
    struct Momentum : public Optimizer {
    public:
        float BETA = 0.9;
        Tensor* m;

        explicit Momentum(float LEARNING_RATE, Parameter* target)
            : Optimizer(LEARNING_RATE, target){
            m = Tensor::declare(target->grad->dims)->create();
        }

        explicit Momentum(float LEARNING_RATE, Parameter* target, float BETA)
            : Momentum(LEARNING_RATE, target){
            this->BETA = BETA;
        }

        void apply() override {
            m = *(*m * BETA) + *target->grad * (1 - BETA);
            target->a = *target->a - *m * LEARNING_RATE;
            *m / LEARNING_RATE;   //we need to recover this
            target->grad->constFill(0);
        }

        void batchApply() override{}
    };


    //AdaGrad : V[t] = sumOf(g[1]^2 ... g[t]^2)
    //          w[t] = w[t-1] - η * g[t] / (sqrt(V[t]) + ε)
    struct AdaGrad : public Optimizer {
    public:
        float EPSILON = 1e-10;
        Tensor* buffer;
        Tensor* gradCopy;

         //A diag matrix with all diag elements value equal to the sum of
         //squared gradients, here stored in float to save for space
        float gVal = 0;

        explicit AdaGrad(float LEARNING_RATE, Parameter* target)
            : Optimizer(LEARNING_RATE, target){
            buffer = target->grad->dims.size < 1025 ? nullptr :
                    Tensor::declare(target->grad->dims.size/1024, 1)->create();
            gradCopy = Tensor::declare(target->grad->dims)->create();
        }

        explicit AdaGrad(float LEARNING_RATE, Parameter* target, float EPSILON)
            : AdaGrad(LEARNING_RATE, target){
            this->EPSILON = EPSILON;
        }

        void apply() override {
            gradCopy->copyD2D(target->grad);
            gVal += reduce(powTensor(gradCopy,2), buffer);
            target->a = *target->a - (*target->grad * (LEARNING_RATE / (sqrt(gVal) + EPSILON)));
            target->grad->constFill(0);
        }

        void batchApply() override{}
    };

    //same as AdaGrad but V[t] = β * V[t-1] + (1 - β) * sum(g[t]^2)
    struct AdaDelta : public AdaGrad {
    public:
        float BETA = 0.99;

        explicit AdaDelta(float LEARNING_RATE, Parameter* target)
            : AdaGrad(LEARNING_RATE, target){}

        explicit AdaDelta(float LEARNING_RATE, Parameter* target, float BETA)
            : AdaGrad(LEARNING_RATE, target), BETA(BETA){}

        void apply() override{
            gradCopy->copyD2D(target->grad);
            gVal = BETA * gVal + (1-BETA) * reduce(powTensor(gradCopy,2), buffer);
            target->a = *target->a - (*target->grad * (LEARNING_RATE / (sqrt(gVal) + EPSILON)));
            target->grad->constFill(0);
        }
    };

    //Adaptive Momentum : m[t] = m[t-1] * β1 + (1 - β1) * g[t]
    //                    V[t] = V[t-1] * β2 + (1 - β2) * g[t]^2
    //                    w[t] = w[t-1] - η * m[t] / (sqrt(V[t]) + ε)
    struct Adam : public AdaGrad {
    public:
        float BETA1 = 0.9;
        float BETA2 = 0.99;
        Tensor* m;

        explicit Adam(float LEARNING_RATE, Parameter* target)
            : AdaGrad(LEARNING_RATE, target){
            m = Tensor::declare(target->grad->dims)->create();
        }

        explicit Adam(float LEARNING_RATE, Parameter* target, float BETA1, float BETA2)
            : AdaGrad(LEARNING_RATE, target), BETA1(BETA1), BETA2(BETA2){
            m = Tensor::declare(target->grad->dims)->create();
        }

        void apply() override {
            m = *(*m * BETA1) + *gradCopy->copyD2D(target->grad) * (1 - BETA1);
            gVal = BETA2 * gVal + (1-BETA2) * reduce(powTensor(gradCopy->copyD2D(target->grad) ,2), buffer);
            target->a = *target->a - (*m * (LEARNING_RATE / (sqrt(gVal) + EPSILON)));
            *m / (LEARNING_RATE / (sqrt(gVal) + EPSILON));  //m is revovered for next iteration
            target->grad->constFill(0);
        }
    };

} // seann

#endif //SEANN_2_OPTIMIZER_CUH
