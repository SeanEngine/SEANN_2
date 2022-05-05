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
        Parameter* A;

        explicit Optimizer(float LEARNING_RATE, Parameter* A)
           : LEARNING_RATE(LEARNING_RATE), A(A){}


        //apply the gradient to the parameters (weights, biases, etc)
        virtual void apply() = 0;

        //updates relative to batches
        virtual void batchApply() = 0;
    };


    //SGD optimizer : Stochastic Gradient Decent, w = w - η * g
    //This optimizer does not have batch behaviors
    struct SGD : public Optimizer {
    public:
        explicit SGD(float LEARNING_RATE, Parameter* A)
            : Optimizer(LEARNING_RATE, A){}

        void apply() override;
        void batchApply() override{}
    };


    //BGD optimizer : Batch Gradient Decent, w = w - (η * sum_batch(g))/bs
    struct BGD : public Optimizer {
    public:
        float BATCH_SIZE;
        explicit BGD(float LEARNING_RATE, Parameter* A, float BATCH_SIZE)
            : Optimizer(LEARNING_RATE, A), BATCH_SIZE(BATCH_SIZE){}

        void apply() override {}

        void batchApply() override;
    };


    //Momentum : m[t] = m[t-1] * β + (1 - β) * g[t]
    //           w[t] = w[t-1] - η * m[t]
    struct Momentum : public Optimizer {
    public:
        float BETA = 0.9;
        Tensor* m;

        explicit Momentum(float LEARNING_RATE, Parameter* A)
            : Optimizer(LEARNING_RATE, A){
            m = Tensor::declare(A->grad->dims)->create();
        }

        explicit Momentum(float LEARNING_RATE, Parameter* A, float BETA)
            : Momentum(LEARNING_RATE, A){
            this->BETA = BETA;
        }

        void apply() override;
        void batchApply() override{}
    };


    //AdaGrad : V[t] = sumOf(g[1]^2 ... g[t]^2)
    //          w[t] = w[t-1] - η * g[t] / (sqrt(V[t]) + ε)
    struct AdaGrad : public Optimizer {
    public:
        float EPSILON = 1e-10;
        Tensor* V;

        explicit AdaGrad(float LEARNING_RATE, Parameter* A)
            : Optimizer(LEARNING_RATE, A){
            V = Tensor::declare(A->grad->dims)->create();
        }

        explicit AdaGrad(float LEARNING_RATE, Parameter* A, float EPSILON)
            : AdaGrad(LEARNING_RATE, A){
            this->EPSILON = EPSILON;
        }

        void apply() override;
        void batchApply() override{}
    };

    //same as AdaGrad but V[t] = β * V[t-1] + (1 - β) * g[t]^2
    //also called RMSProp
    struct AdaDelta : public AdaGrad {
    public:
        float BETA = 0.99;

        explicit AdaDelta(float LEARNING_RATE, Parameter* A)
            : AdaGrad(LEARNING_RATE, A){}

        explicit AdaDelta(float LEARNING_RATE, Parameter* A, float BETA)
            : AdaGrad(LEARNING_RATE, A), BETA(BETA){}

        void apply() override;
    };

    //Adaptive Momentum : m[t] = m[t-1] * β1 + (1 - β1) * g[t]
    //                    V[t] = V[t-1] * β2 + (1 - β2) * g[t]^2
    //                    w[t] = w[t-1] - η * m[t] / (sqrt(V[t]) + ε)
    struct Adam : public AdaGrad {
    public:
        float BETA1 = 0.9;
        float BETA2 = 0.99;
        Tensor* m;

        explicit Adam(float LEARNING_RATE, Parameter* A)
            : AdaGrad(LEARNING_RATE, A){
            m = Tensor::declare(A->grad->dims)->create();
        }

        explicit Adam(float LEARNING_RATE, Parameter* A, float BETA1, float BETA2)
            : AdaGrad(LEARNING_RATE, A), BETA1(BETA1), BETA2(BETA2){
            m = Tensor::declare(A->grad->dims)->create();
        }

        void apply() override;
    };

} // seann

#endif //SEANN_2_OPTIMIZER_CUH
