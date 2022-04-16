cmake_minimum_required(VERSION 3.20)
project(SEANN_2 CUDA CXX)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)

set(OpenCV_DIR D:\\OpenCV_CUDA_4.5.4\\build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(SEANN_2 main.cu seblas/tensor/Tensor.cu seblas/tensor/Tensor.cuh seblas/assist/TensorAssert.cu
        seblas/assist/TensorAssert.cuh seio/logging/LogUtils.cuh seio/logging/LogUtils.cu seio/logging/Color.cuh
        seio/logging/Color.cu seblas/assist/CudaAssert.cu seblas/assist/CudaAssert.cuh seutil/exec/ThreadController.cu
        seutil/exec/ThreadController.cuh)

set_target_properties(SEANN_2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 86)
