cmake_minimum_required(VERSION 3.20)
project(2D_Stencil CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)

get_filename_component(CUDA_INSTALL_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
get_filename_component(CUDA_INSTALL_DIR ${CUDA_INSTALL_DIR} DIRECTORY)

include_directories("${CUDA_INSTALL_DIR}/samples/common/inc/")

add_link_options("-fopenmp")

add_executable(
        2D_Stencil
        src/main.cu
        src/single-gpu-naive.cu include/single-gpu-naive.cuh
        src/multi-gpu-peer.cu include/multi-gpu-peer.cuh
        src/common.cu include/common.h
        src/single-stream/multi-threaded.cu include/single-stream/multi-threaded.cuh
        src/single-stream/single-threaded.cu include/single-stream/single-threaded.cuh
        src/single-stream/multi-threaded-two-block-comm.cu include/single-stream/multi-threaded-two-block-comm.cuh
        src/single-stream/single-threaded-two-block-comm.cu include/single-stream/single-threaded-two-block-comm.cuh)

set_target_properties(2D_Stencil PROPERTIES CUDA_ARCHITECTURES "70;80")

set_target_properties(2D_Stencil PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
