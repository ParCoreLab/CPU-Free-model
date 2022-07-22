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
        src/common.cu include/common.h

        src/baseline/multi-threaded-copy-overlap.cu include/baseline/multi-threaded-copy-overlap.cuh
        src/baseline/multi-threaded-copy.cu include/baseline/multi-threaded-copy.cuh
        src/baseline/single-threaded-copy.cu include/baseline/single-threaded-copy.cuh
        src/baseline/multi-threaded-p2p.cu include/baseline/multi-threaded-p2p.cuh
        src/baseline/nvshmem.cu include/baseline/nvshmem.cuh
        src/baseline/nvshmem_opt.cu include/baseline/nvshmem_opt.cuh

        src/multi-stream/multi-gpu-peer-tiling.cu include/multi-stream/multi-gpu-peer-tiling.cuh
        src/multi-stream/multi-gpu-peer-tiling-half.cu include/multi-stream/multi-gpu-peer-tiling-half.cuh

        src/single-stream/multi-threaded-two-block-comm.cu include/single-stream/multi-threaded-two-block-comm.cuh
        src/single-stream/multi-threaded-one-block-comm.cu include/single-stream/multi-threaded-one-block-comm.cuh

        src/no-compute/multi-gpu-peer-tiling-no-compute.cu include/no-compute/multi-gpu-peer-tiling-no-compute.cuh
        src/no-compute/multi-threaded-copy-no-compute.cu include/no-compute/multi-threaded-copy-no-compute.cuh
        src/no-compute/multi-threaded-copy-overlap-no-compute.cu include/no-compute/multi-threaded-copy-overlap-no-compute.cuh
        src/no-compute/multi-threaded-one-block-comm-no-compute.cu include/no-compute/multi-threaded-one-block-comm-no-compute.cuh
        src/no-compute/multi-threaded-p2p-no-compute.cu include/no-compute/multi-threaded-p2p-no-compute.cuh
        src/no-compute/multi-threaded-two-block-comm-no-compute.cu include/no-compute/multi-threaded-two-block-comm-no-compute.cuh)

set_target_properties(2D_Stencil PROPERTIES CUDA_ARCHITECTURES "70;80")

set_target_properties(2D_Stencil PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
