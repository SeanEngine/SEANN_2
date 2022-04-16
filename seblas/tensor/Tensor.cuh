//
// Created by Wake on 4/16/2022.
//

#ifndef SEANN_2_TENSOR_CUH
#define SEANN_2_TENSOR_CUH

#include "cuda.h"
#include "cuda_runtime.h"
#include "mma.h"
#include "curand.h"
#include <curand_kernel.h>

#include <string>
#include <cassert>
#include <random>

#include "../assist/CudaAssert.cuh"
#include "../../seutil/exec/ThreadController.cuh"

#define CPU_THREADS 12


using namespace std;
using namespace nvcuda;
using namespace seutil;
namespace seblas{

    const dim3 CUDA_BLOCK_SIZE = dim3(16, 16);
    typedef unsigned int uint32;

    /**
     * @Brief index4 is used to navigate within a 4D tensor
     * Tensors are stored in row major order (NCHW)
     *
     * //NHWC is not supported because its a cursed arrangement
     * that should be burned to death in the flame of hell
     *
     * n : the 4th dimension
     * c : channels
     * h : rows (height)
     * w : cols (width)
     */
    struct index4{
        uint32 n=0, c=0, h=0, w=0;
        __device__ __host__ index4(uint32 n, uint32 c, uint32 h, uint32 w) : n(n), c(c), h(h), w(w){}
        __device__ __host__ index4(uint32 c, uint32 h, uint32 w) : c(c), h(h), w(w){}
        __device__ __host__ index4(uint32 h, uint32 w) : h(h), w(w){}

        __device__ __host__ index4 operator+(index4 other) const;
        __device__ __host__ index4 operator-(index4 other) const;
        __device__ __host__ bool operator==(index4 other) const;
        __device__ __host__ bool operator>(index4 other) const;
        __device__ __host__ bool operator<(index4 other) const;

        [[nodiscard]] __device__ __host__ uint32 getOffset() const;
        [[nodiscard]] string toString() const;
    };

    /**
     * shape of tensors
     */
    struct shape4 : public index4{
        uint32 size;
        __device__ __host__ shape4(uint32 n, uint32 c, uint32 h, uint32 w)
            : index4(n, c, h, w){ size = n*c*h*w; }

        __device__ __host__ shape4(uint32 c, uint32 h, uint32 w)
            : index4(1, c, h, w){ size = c*h*w; }

        __device__ __host__ shape4(uint32 h, uint32 w)
            : index4(1, 1, h, w){ size = h*w; }
    };

    /**
     * Tensor is the base of everything,
     * it records the dimensions and a pointer to data elements
     * Tensor supports FP32 and TF32 as data types
     */
    class Tensor {
    public:
        shape4 dims;
        float* elements = nullptr;
        int deviceId = 0;

        static Tensor* declare(shape4 dims) {
            Tensor *construct;
            cudaMallocHost(&construct, sizeof(Tensor));
            construct->dims = dims;
            return construct;
        }

        template<typename... Args>
        static Tensor* declare(Args &&... args) {
            return declare(shape4(std::forward<Args>(args)...));
        }

        //create and destruct tensor elements
        Tensor* create();
        Tensor* destroy();
        void eliminate();
        Tensor* createHost();
        Tensor* destroyHost();
        void eliminateHost();

        //transLocate
        //toDevice() and toHost() will migrate the elements
        //the original tensor would be unregistered
        //ripoff() creates an identical tensor on host as it is on device
        Tensor* toDevice();
        Tensor* toHost();
        Tensor* ripOffDevice() const;
        Tensor* copyH2D(Tensor* onDevice) const;
        Tensor* copyD2H(Tensor* onHost) const;
        Tensor* copyD2D(Tensor* onDevice) const;
        Tensor* copyH2H(Tensor* onHost) const;

        //attaching (Tensors sharing same elements)
        Tensor* attach(Tensor* other);
        Tensor* attach(float* element, int deviceID);

        //common operators
        Tensor* operator+(Tensor* other);
        Tensor* operator+(float other);
        Tensor* operator-(Tensor* other);
        Tensor* operator-(float other);
        Tensor* operator*(Tensor* other);  //hadamard product
        Tensor* operator*(float other);
        Tensor* operator/(Tensor* other);
        Tensor* operator/(float other);

        //initialization
        Tensor* constFill(float val);
        Tensor* randUniform(float min, float max);
        Tensor* randNormal(float mean, float stddev);
    };
}


#endif //SEANN_2_TENSOR_CUH
