cmake_minimum_required(VERSION 3.15)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_STANDARD 17)
project(cuda_gaussian_blur CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CUDA_GAUSSIAN_SRC
        src/convolution_kernel.cu
        src/gaussian_blur.cu)

add_library(cuda_gaussian_blur ${CUDA_GAUSSIAN_SRC})
target_include_directories(cuda_gaussian_blur PUBLIC
        include
        ${CUDA_INCLUDE_DIRS}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        )

set_target_properties(cuda_gaussian_blur PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
