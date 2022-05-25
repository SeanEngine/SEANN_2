cmake_minimum_required(VERSION 3.22)
set(CMAKE_CUDA_COMPILER D:/Program\ Files/NVIDIA/CUDA\ Development/bin)
project(SEANN_2 CUDA CXX)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)


set(OpenCV_DIR D:\\Dependencies\\OpenCV_CUDA\\build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(SEANN_2
        main.cu
        seio/logging/LogUtils.cuh
        seio/logging/LogUtils.cu
        seio/logging/Color.cuh
        seio/logging/Color.cu

        seblas/tensor/Tensor.cu
        seblas/tensor/Tensor.cuh
        seblas/assist/TensorAssert.cu
        seblas/assist/TensorAssert.cuh
        seblas/assist/CudaAssert.cu
        seblas/assist/CudaAssert.cuh
        seblas/assist/Inspection.cu
        seblas/assist/Inspection.cuh
        seutil/exec/ThreadController.cu
        seutil/exec/ThreadController.cuh
        seblas/operations/cuGEMM.cu
        seblas/operations/cuGEMM.cuh
        seblas/operations/cuConv.cu
        seblas/operations/cuConv.cuh
        seblas/operations/cuActivation.cu
        seblas/operations/cuActivation.cuh
        seblas/operations/cuReduce.cu
        seblas/operations/cuReduce.cuh
        seblas/operations/cuOperations.cuh
        seblas/tensor/Assistance.cu
        seblas/tensor/Assistance.cuh

        seann/operands/Linear.cu
        seann/operands/Linear.cuh
        seann/operands/OperandBase.cu
        seann/operands/OperandBase.cuh
        seann/components/Parameter.cu
        seann/components/Parameter.cuh
        seann/optimizers/Optimizer.cu
        seann/optimizers/Optimizer.cuh
        seann/components/NetParam.cu
        seann/components/NetParam.cuh
        seann/operands/ReLU.cu
        seann/operands/ReLU.cuh
        seann/operands/Conv2D.cu
        seann/operands/Conv2D.cuh
        seann/operands/Softmax.cu
        seann/operands/Softmax.cuh
        seann/containers/Sequential.cu
        seann/containers/Sequential.cuh
        seann/seann.cuh
        seann/optimizers/Loss.cuh
        seann/optimizers/Loss.cu
        seio/data/Dataset.cu
        seio/data/Dataset.cuh
        seio/data/DataLoader.cu
        seio/data/DataLoader.cuh
        seann/operands/MaxPool2D.cu
        seann/operands/MaxPool2D.cuh
        seblas/operations/cuParallel.cu
        seblas/operations/cuParallel.cuh)

set_target_properties(SEANN_2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 86)
