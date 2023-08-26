cmake_minimum_required(VERSION 3.23)

set(CMAKE_VERBOSE_MAKEFILE ON)

project(cpu-free-model VERSION 1.0
        DESCRIPTION "TBD"
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(NVCC_VER_MIN 11.6)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# CUDA
set(CMAKE_CUDA_ARCHITECTURES "80")
find_package(CUDAToolkit ${NVCC_VER_MIN} REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

enable_language(CUDA)

# Why is this necessary
find_package(OpenMP REQUIRED)
list(APPEND CMAKE_CUDA_FLAGS "-Xcompiler -fopenmp")

add_subdirectory(CG)
add_subdirectory(Stencil)
