cmake_minimum_required(VERSION 3.10)
project(2D_Stencil CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)

get_filename_component(CUDA_INSTALL_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
get_filename_component(CUDA_INSTALL_DIR ${CUDA_INSTALL_DIR} DIRECTORY)

include_directories("${CUDA_INSTALL_DIR}/samples/common/inc/")

#add_link_options("-fopenmp")

add_executable(
        2D_Stencil
        src/main.cu
        #        src/single-gpu-naive.cu include/single-gpu-naive.cuh
        src/multi-gpu-peer.cu include/multi-gpu-peer.cuh
)

add_executable(
        single_stream_single_threaded-jacobi
        src/main.cu
        src/single-stream/jacobi-single-threaded.cu)

add_executable(
        single_stream_multi_threaded-jacobi
        src/main.cu
        src/single-stream/jacobi-multi-threaded.cu)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")

set_target_properties(2D_Stencil PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
