add_executable(jacobi2D
        src/main.cu
        src/common.cu
        PERKS/jacobi-general-wrapper.cu
        src/baseline/multi-threaded-copy.cu
        src/baseline/multi-threaded-copy-overlap.cu
        src/baseline/multi-threaded-p2p.cu
        src/baseline/single-threaded-copy.cu
        src/multi-stream/multi-gpu-peer-tiling.cu
        src/PERKS/multi-stream-perks.cu
        src/single-stream/multi-threaded-one-block-comm.cu
        src/single-stream/multi-threaded-two-block-comm.cu
        src/no-compute/multi-gpu-peer-tiling-no-compute.cu
        src/no-compute/multi-threaded-copy-no-compute.cu
        src/no-compute/multi-threaded-copy-overlap-no-compute.cu
        src/no-compute/multi-threaded-one-block-comm-no-compute.cu
        src/no-compute/multi-threaded-p2p-no-compute.cu
        src/no-compute/multi-threaded-two-block-comm-no-compute.cu)

add_executable(jacobi2D_nvshmem
        src_nvshmem/main.cu
        src_nvshmem/common.cu
        PERKS/jacobi-general-wrapper.cu
        src_nvshmem/baseline/multi-threaded-nvshmem.cu
        src_nvshmem/baseline/multi-threaded-nvshmem-opt.cu
        src_nvshmem/PERKS/multi-stream-perks.cu
        src_nvshmem/multi-stream/multi-gpu-multi-block-tiling.cu
        src_nvshmem/multi-stream/multi-gpu-peer-tiling.cu
        src_nvshmem/single-stream/multi-threaded-multi-block-comm.cu
        src_nvshmem/single-stream/multi-threaded-one-block-comm.cu
        src_nvshmem/single-stream/multi-threaded-two-block-comm.cu
        src_nvshmem/no-compute/design-1-multi-block-no-compute.cu
        src_nvshmem/no-compute/multi-gpu-peer-tiling-no-compute.cu
        src_nvshmem/no-compute/multi-threaded-nvshmem-no-compute.cu
        src_nvshmem/no-compute/multi-threaded-nvshmem-opt-no-compute.cu
        src_nvshmem/no-compute/multi-threaded-one-block-comm-no-compute.cu
        src_nvshmem/no-compute/multi-threaded-two-block-comm-no-compute.cu)

target_include_directories(jacobi2D PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(jacobi2D PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/PERKS")

target_include_directories(jacobi2D_nvshmem PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include_nvshmem")
target_include_directories(jacobi2D_nvshmem PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/PERKS")

find_package(OpenMP REQUIRED)
find_package(NVSHMEM REQUIRED)
find_package(MPI REQUIRED)

target_link_libraries(jacobi2D
        CUDA::cudart
        OpenMP::OpenMP_CXX)

target_link_libraries(jacobi2D_nvshmem
        CUDA::cudart
        OpenMP::OpenMP_CXX
        nvshmem::nvshmem
        MPI::MPI_CXX)
