add_executable(cg
        src/main.cu
        src/common.cu
        src/mmio.c
        src/mmio_wrapper.cpp
        src/baseline/discrete-pipelined.cu
        src/baseline/discrete-standard.cu
        src/profiling/discrete-pipelined.cu
        src/profiling/discrete-standard.cu
        src/single-gpu/discrete-standard.cu
        src/single-stream/pipelined.cu
        src/single-stream/pipelined-gather.cu
        src/single-stream/pipelined-multi-overlap.cu
        src/single-stream/standard.cu
        src/single-stream/standard-saxpy-overlap.cu)

find_package(OpenMP REQUIRED)
find_package(NVSHMEM REQUIRED)
find_package(MPI REQUIRED)

target_link_libraries(cg
        CUDA::cudart
        OpenMP::OpenMP_CXX
        nvshmem::nvshmem
        MPI::MPI_CXX)

target_include_directories(cg SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_compile_options(cg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -D_FORCE_INLINES
        >)
