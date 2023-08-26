add_executable(jacobi3D
        src/main.cu
        src/common.cu
        src/baseline/multi-threaded-copy.cu
        src/baseline/multi-threaded-copy-overlap.cu
        src/baseline/multi-threaded-p2p.cu
        src/baseline/single-threaded-copy.cu
        src/multi-stream/multi-gpu-peer-tiling.cu
        src/single-stream/multi-threaded-one-block-comm.cu
        src/single-stream/multi-threaded-two-block-comm.cu
        src/PERKS/common/common.hpp
        src/PERKS/common/cub_utils.cuh
        src/PERKS/common/cuda_common.cuh
        src/PERKS/common/cuda_computation.cuh
        src/PERKS/common/jacobi_cuda.cuh
        src/PERKS/common/jacobi_reference.hpp
        src/PERKS/common/types.hpp
        src/PERKS/config.cuh
        src/PERKS/genconfig.cuh
        src/PERKS/j3d-general-kernels.cuh
        src/PERKS/j3d-general-wrapper.cu
        src/PERKS/multi-stream-perks.cu
        src/PERKS/perksconfig.cuh
        src/no-compute/multi-gpu-peer-tiling-no-compute.cu
        src/no-compute/multi-threaded-copy-no-compute.cu
        src/no-compute/multi-threaded-copy-overlap-no-compute.cu
        src/no-compute/multi-threaded-one-block-comm-no-compute.cu
        src/no-compute/multi-threaded-p2p-no-compute.cu
        src/no-compute/multi-threaded-two-block-comm-no-compute.cu)

add_executable(jacobi3D_nvshmem
        src_nvshmem/baseline/multi-threaded-nvshmem.cu
        src_nvshmem/baseline/multi-threaded-nvshmem-opt.cu
        src_nvshmem/common.cu
        src_nvshmem/main.cu
        src_nvshmem/multi-stream/multi-gpu-multi-block-tiling.cu
        src_nvshmem/multi-stream/multi-gpu-peer-tiling.cu
        src_nvshmem/no-compute/multi-gpu-peer-tiling-no-compute.cu
        src_nvshmem/no-compute/multi-threaded-multi-block-comm-no-compute.cu
        src_nvshmem/no-compute/multi-threaded-nvshmem-no-compute.cu
        src_nvshmem/no-compute/multi-threaded-nvshmem-opt-no-compute.cu
        src_nvshmem/no-compute/multi-threaded-one-block-comm-no-compute.cu
        src_nvshmem/no-compute/multi-threaded-two-block-comm-no-compute.cu
        src_nvshmem/PERKS-nvshmem/common/common.hpp
        src_nvshmem/PERKS-nvshmem/common/cub_utils.cuh
        src_nvshmem/PERKS-nvshmem/common/cuda_common.cuh
        src_nvshmem/PERKS-nvshmem/common/cuda_computation.cuh
        src_nvshmem/PERKS-nvshmem/common/jacobi_cuda.cuh
        src_nvshmem/PERKS-nvshmem/common/jacobi_reference.hpp
        src_nvshmem/PERKS-nvshmem/common/types.hpp
        src_nvshmem/PERKS-nvshmem/config.cuh
        src_nvshmem/PERKS-nvshmem/genconfig.cuh
        src_nvshmem/PERKS-nvshmem/j3d-general-kernels.cuh
        src_nvshmem/PERKS-nvshmem/j3d-general-wrapper.cu
        src_nvshmem/PERKS-nvshmem/multi-stream-perks-block.cu
        src_nvshmem/PERKS-nvshmem/multi-stream-perks.cu
        src_nvshmem/PERKS-nvshmem/perksconfig.cuh
        src_nvshmem/single-stream/multi-threaded-multi-block-comm.cu
        src_nvshmem/single-stream/multi-threaded-one-block-comm.cu
        src_nvshmem/single-stream/multi-threaded-two-block-comm.cu)

target_include_directories(jacobi3D PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(jacobi3D_nvshmem PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include_nvshmem")

find_package(OpenMP REQUIRED)
find_package(NVSHMEM REQUIRED)
find_package(MPI REQUIRED)

target_link_libraries(jacobi3D
        CUDA::cudart
        OpenMP::OpenMP_CXX)

target_link_libraries(jacobi3D_nvshmem
        CUDA::cudart
        OpenMP::OpenMP_CXX
        nvshmem::nvshmem
        MPI::MPI_CXX)
