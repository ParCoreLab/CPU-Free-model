cmake_minimum_required(VERSION 3.20)
project(3D_Stencil CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)

get_filename_component(CUDA_INSTALL_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
get_filename_component(CUDA_INSTALL_DIR ${CUDA_INSTALL_DIR} DIRECTORY)

include_directories("${CUDA_INSTALL_DIR}/samples/common/inc/")

add_link_options("-fopenmp")

add_executable(
        3D_Stencil
        src/main.cu
        src/common.cu include/common.h
        src/multi-stream/multi-gpu-peer.cu include/multi-stream/multi-gpu-peer.cuh
        src/single-stream/multi-threaded.cu include/single-stream/multi-threaded.cuh
        src/single-stream/single-threaded.cu include/single-stream/single-threaded.cuh
        src/single-stream/multi-threaded-two-block-comm.cu include/single-stream/multi-threaded-two-block-comm.cuh
        src/single-stream/single-threaded-two-block-comm.cu include/single-stream/single-threaded-two-block-comm.cuh)

set_target_properties(3D_Stencil PROPERTIES CUDA_ARCHITECTURES "70;80")

set_target_properties(3D_Stencil PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
