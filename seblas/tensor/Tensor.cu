//
// Created by Wake on 4/16/2022.
//

#include "Tensor.cuh"
#define toFloat4R(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
#define CUDA_BLOCK_SIZE_1D CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y
#define topOff(a,b) (a + b - 1)/(b)


namespace seblas {

    //CPU operators
    void addTensorCPU(const int tid, const int ts, Tensor* A, Tensor* B){
        int start = ts * (int)(A->dims.size/tid);
        int end = ts == tid-1 ? (int)A->dims.size : start + (int)A->dims.size/tid;
        for(int i = start; i < end; i++){
            A->elements[i] += B->elements[i];
        }
    }

    void minusTensorCPU(const int tid, const int ts, Tensor* A, Tensor* B){
        int start = ts * (int)(A->dims.size/tid);
        int end = ts == tid-1 ? (int)A->dims.size : start + (int)A->dims.size/tid;
        for(int i = start; i < end; i++){
            A->elements[i] -= B->elements[i];
        }
    }

    void hadamardTensorCPU(const int tid, const int ts, Tensor* A, Tensor* B){
        int start = ts * (int)(A->dims.size/tid);
        int end = ts == tid-1 ? (int)A->dims.size : start + (int)A->dims.size/tid;
        for(int i = start; i < end; i++){
            A->elements[i] *= B->elements[i];
        }
    }

    void divideTensorCPU(const int tid, const int ts, Tensor* A, Tensor* B){
        int start = ts * (int)(A->dims.size/tid);
        int end = ts == tid-1 ? (int)A->dims.size : start + (int)A->dims.size/tid;
        for(int i = start; i < end; i++){
            A->elements[i] /= B->elements[i];
        }
    }

    //GPU operators
    __global__ void addD(Tensor* in, Tensor* other){
        uint32 index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < in->dims.size){
            in->elements[index] += other->elements[index];
        }
    }

    __global__ void add4D(Tensor* in, Tensor* other){
        uint32 index = (threadIdx.x + blockIdx.x * blockDim.x )* 4;
        float regisA[4];
        float regisB[4];
        float regisC[4] = {0};
        if(index < in->dims.size){
            toFloat4R(regisA[0]) = toFloat4R(in->elements[index]);
            toFloat4R(regisB[0]) = toFloat4R(other->elements[index]);
            #pragma unroll
            for (int i = 0; i < 4; i++){
                regisC[i] = regisA[i] + regisB[i];
            }
            toFloat4R(in->elements[index]) = toFloat4R(regisC[0]);
        }
    }

    __global__ void subtractD(Tensor* A, Tensor* B){
        uint32 index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < A->dims.size){
            A->elements[index] -= B->elements[index];
        }
    }

    __global__ void subtract4D(Tensor* A, Tensor* B){
        uint32 index = (threadIdx.x + blockIdx.x * blockDim.x )* 4;
        float regisA[4];
        float regisB[4];
        float regisC[4] = {0};
        if(index < A->dims.size){
            toFloat4R(regisA[0]) = toFloat4R(A->elements[index]);
            toFloat4R(regisB[0]) = toFloat4R(B->elements[index]);
            #pragma unroll
            for (int i = 0; i < 4; i++){
                regisC[i] = regisA[i] - regisB[i];
            }
            toFloat4R(A->elements[index]) = toFloat4R(regisC[0]);
        }
    }

    __global__ void hadamardProductD(Tensor* in, Tensor* other){
        uint32 index = threadIdx.y + blockIdx.y * blockDim.y;
        if(index < in->dims.size){
            in->elements[index] *= other->elements[index];
        }
    }

    __global__ void hadamardProduct4D(Tensor* in, Tensor* other){
        uint32 index = (threadIdx.x + blockIdx.x * blockDim.x )* 4;
        float regisA[4];
        float regisB[4];
        float regisC[4] = {0};
        if(index < in->dims.size){
            toFloat4R(regisA[0]) = toFloat4R(in->elements[index]);
            toFloat4R(regisB[0]) = toFloat4R(other->elements[index]);
            #pragma unroll
            for (int i = 0; i < 4; i++){
                regisC[i] = regisA[i] * regisB[i];
            }
            toFloat4R(in->elements[index]) = toFloat4R(regisC[0]);
        }
    }

    __global__ void divideD(Tensor* in, Tensor* other){
        uint32 index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < in->dims.size){
            in->elements[index] /= other->elements[index];
        }
    }
    
    __global__ void divide4D(Tensor* in, Tensor* other){
        uint32 index = (threadIdx.x + blockIdx.x * blockDim.x )* 4;
        float regisA[4];
        float regisB[4];
        float regisC[4] = {0};
        if(index < in->dims.size){
            toFloat4R(regisA[0]) = toFloat4R(in->elements[index]);
            toFloat4R(regisB[0]) = toFloat4R(other->elements[index]);
            #pragma unroll
            for (int i = 0; i < 4; i++){
                regisC[i] = regisA[i] / regisB[i];
            }
            toFloat4R(in->elements[index]) = toFloat4R(regisC[0]);
        }
    }

    //index operators
    __device__ __host__ seblas::index4 seblas::index4::operator+(seblas::index4 other) const {
        return {n + other.n, c + other.c, h + other.h, w + other.w};
    }

    __device__ __host__ index4 index4::operator-(index4 other) const {
        return {n - other.n, c - other.c, h - other.h, w - other.w};
    }

    __device__ __host__ bool index4::operator==(index4 other) const {
        return n == other.n && c == other.c && h == other.h && w == other.w;
    }

    __device__ __host__ bool index4::operator>(index4 other) const {
        return n > other.n && c > other.c && h > other.h && w > other.w;
    }

    __device__ __host__ bool index4::operator<(index4 other) const {
        return n < other.n && c < other.c && h < other.h && w < other.w;
    }

    __device__ __host__ uint32 index4::getOffset() const {
        return n == 0 ? 1 : n * c == 0 ? 1 : c * h == 0 ? 1 : h * w;
    }

    string index4::toString() const {
        return "(" + to_string(n) + "," + to_string(c) + "," + to_string(h) + "," + to_string(w) + ")";
    }

    Tensor *Tensor::create() {
        cudaMalloc(&elements, sizeof(float) * dims.size);
        assertCuda(__FILE__, __LINE__);
        deviceId = 0;
        return this;
    }

    Tensor *Tensor::destroy() {
        cudaFree(elements);
        assertCuda(__FILE__, __LINE__);
        return this;
    }

    Tensor *Tensor::createHost() {
        cudaMallocHost(&elements, sizeof(float) * dims.size);
        assertCuda(__FILE__, __LINE__);
        deviceId = -1;
        return this;
    }

    Tensor *Tensor::destroyHost() {
        cudaFreeHost(elements);
        assertCuda(__FILE__, __LINE__);
        return this;
    }

    Tensor *Tensor::toDevice() {
        auto* output = Tensor::declare(dims)->create();
        copyH2D(output);
        destroyHost();
        cudaFreeHost(this);
        return output;
    }

    Tensor *Tensor::toHost() {
        auto* output = Tensor::declare(dims)->createHost();
        copyD2H(output);
        destroy();
        cudaFreeHost(this);
        return output;
    }

    Tensor *Tensor::copyH2D(Tensor *onDevice) const {
        assert(dims.size == onDevice->dims.size);
        cudaMemcpy(onDevice->elements, elements, sizeof(float) * dims.size, cudaMemcpyHostToDevice);
        assertCuda(__FILE__, __LINE__);
        return onDevice;
    }

    Tensor *Tensor::copyD2H(Tensor *onHost) const {
        assert(dims.size == onHost->dims.size);
        cudaMemcpy(onHost->elements, elements, sizeof(float) * dims.size, cudaMemcpyDeviceToHost);
        assertCuda(__FILE__, __LINE__);
        return onHost;
    }

    Tensor *Tensor::copyD2D(Tensor *onDevice) const {
        assert(dims.size == onDevice->dims.size);
        cudaMemcpy(onDevice->elements, elements, sizeof(float) * dims.size, cudaMemcpyDeviceToDevice);
        assertCuda(__FILE__, __LINE__);
        return onDevice;
    }

    Tensor *Tensor::copyH2H(Tensor *onHost) const {
        assert(dims.size == onHost->dims.size);
        cudaMemcpy(onHost->elements, elements, sizeof(float) * dims.size, cudaMemcpyHostToHost);
        assertCuda(__FILE__, __LINE__);
        return onHost;
    }

    Tensor *Tensor::attach(Tensor *other) {
        assert(dims.size <= other->dims.size);
        elements = other->elements;
        deviceId = other->deviceId;
        return this;
    }

    Tensor *Tensor::attach(float *other, int deviceID) {
        elements = other;
        deviceId = deviceID;
        return this;
    }

    Tensor *Tensor::operator+(Tensor *other) {
        assert(dims.size == other->dims.size);
        assert(deviceId == other->deviceId);
        if(deviceId == -1){
            _alloc<CPU_THREADS>(addTensorCPU, this, other);
            return this;
        }

        if(dims.size % 4 == 0){
            uint32 grid = topOff(dims.size/4, CUDA_BLOCK_SIZE_1D);
            uint32 block = CUDA_BLOCK_SIZE_1D;
            add4D<<<grid, block>>>(this, other);
            assertCuda(__FILE__, __LINE__);
            return this;
        }

        uint32 grid = topOff(dims.size, CUDA_BLOCK_SIZE_1D);
        uint32 block = CUDA_BLOCK_SIZE_1D;
        addD<<<grid, block>>>(this, other);
        assertCuda(__FILE__, __LINE__);
        return this;
    }

    Tensor *Tensor::operator-(Tensor *other) {
        assert(dims.size == other->dims.size);
        assert(deviceId == other->deviceId);
        if(deviceId == -1){
            _alloc<CPU_THREADS>(minusTensorCPU, this, other);
            return this;
        }

        if(dims.size % 4 == 0){
            uint32 grid = topOff(dims.size/4, CUDA_BLOCK_SIZE_1D);
            uint32 block = CUDA_BLOCK_SIZE_1D;
            subtract4D<<<grid, block>>>(this, other);
            assertCuda(__FILE__, __LINE__);
            return this;
        }

        uint32 grid = topOff(dims.size, CUDA_BLOCK_SIZE_1D);
        uint32 block = CUDA_BLOCK_SIZE_1D;
        subtractD<<<grid, block>>>(this, other);
        assertCuda(__FILE__, __LINE__);
        return this;
    }

    Tensor *Tensor::operator*(Tensor *other) {
        assert(dims.size == other->dims.size);
        assert(deviceId == other->deviceId);
        if(deviceId == -1){
            _alloc<CPU_THREADS>(hadamardTensorCPU, this, other);
            return this;
        }

        if(dims.size % 4 == 0){
            uint32 grid = topOff(dims.size/4, CUDA_BLOCK_SIZE_1D);
            uint32 block = CUDA_BLOCK_SIZE_1D;
            hadamardProduct4D<<<grid, block>>>(this, other);
            assertCuda(__FILE__, __LINE__);
            return this;
        }

        uint32 grid = topOff(dims.size, CUDA_BLOCK_SIZE_1D);
        uint32 block = CUDA_BLOCK_SIZE_1D;
        hadamardProductD<<<grid, block>>>(this, other);
        assertCuda(__FILE__, __LINE__);
        return this;
    }

    Tensor *Tensor::operator/(Tensor *other) {
        assert(dims.size == other->dims.size);
        assert(deviceId == other->deviceId);
        if(deviceId == -1){
            _alloc<CPU_THREADS>(divideTensorCPU, this, other);
            return this;
        }

        if(dims.size % 4 == 0){
            uint32 grid = topOff(dims.size/4, CUDA_BLOCK_SIZE_1D);
            uint32 block = CUDA_BLOCK_SIZE_1D;
            divide4D<<<grid, block>>>(this, other);
            assertCuda(__FILE__, __LINE__);
            return this;
        }

        uint32 grid = topOff(dims.size, CUDA_BLOCK_SIZE_1D);
        uint32 block = CUDA_BLOCK_SIZE_1D;
        divideD<<<grid, block>>>(this, other);
        assertCuda(__FILE__, __LINE__);
        return this;
    }
}
