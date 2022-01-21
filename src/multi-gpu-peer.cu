/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <iterator>

#include <omp.h>

#include <cooperative_groups.h>

#include "../include/common.h"
#include "../include/single-gpu-naive.cuh"

namespace cg = cooperative_groups;

constexpr real zeroTwentyFive { 0.25 };
const real PI { static_cast<real>(2.0 * std::asin(1.0)) };

__global__ void initialize_boundaries(real* __restrict__ const a_new, real* __restrict__ const a,
                                      const real pi, const int nx, const int ny) {
    for (unsigned int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < ny; iy += blockDim.x * gridDim.x) {
        const real y0 = sin(2.0 * pi * iy / (ny - 1));
        a[iy * nx + 0] = y0;
        a[iy * nx + (nx - 1)] = y0;
        a_new[iy * nx + 0] = y0;
        a_new[iy * nx + (nx - 1)] = y0;
    }
}

__global__ void jacobi_kernel(real* __restrict__ a_new, const real* __restrict__ a,
                              const int iy_start, const int iy_end, const int nx, const int niter,
                              int* flag) {

    cg::grid_group grid = cg::this_grid();

    const unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;

    int i = 0;

    while (i < niter) {
        if (iy < iy_end) {
            if (ix >= 1 && ix < (nx - 1)) {
                const real new_val = zeroTwentyFive * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                             a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
                a_new[iy * nx + ix] = new_val;

                // apply boundary conditions
                if (iy_start == iy) {
                    a_new[iy_end * nx + ix] = new_val;
                }

                if ((iy_end - 1) == iy) {
                    a_new[(iy_start - 1) * nx + ix] = new_val;
                }
            }
        }

        real* temp_pointer = a_new;
        a = a_new;
        a_new = temp_pointer;

        i++;
        grid.sync();
    }

    if (threadIdx.x == 0) {
        *flag = 1;
    }
}

__global__ void boundary_sync_kernel(real* __restrict__ a_new, int* flag) {
    while (!*flag) {
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("Sync\n");
        *flag = false;
    }
}

bool get_arg(char** begin, char** end, const std::string& arg) {
    char** itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

constexpr int THREADS_PER_BLOCK = 1024;

int init(int argc, char* argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nccheck = get_argval<int>(argv, argv + argc, "-nccheck", 1);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);
    const bool csv = get_arg(argv, argv + argc, "-csv");

    if (nccheck != 1) {
        fprintf(stderr, "Only nccheck = 1 is supported\n");
        return -1;
    }

    real* a;
    real* a_new;

    int iy_start = 1;
    int iy_end = (ny - 1);

    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

    // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp{};
    int devID = 0;  // findCudaDevice(argc, (const char **)argv);
    CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, devID));
    int numSms = deviceProp.multiProcessorCount;

    int numBlocksPerSm = 0;
    int numThreads = THREADS_PER_BLOCK;

    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, jacobi_kernel,
                                                               numThreads, 0));

    // This is stupid
    int blocks_each = (int)sqrt(numSms * numBlocksPerSm);
    int threads_each = (int)sqrt(THREADS_PER_BLOCK);
    dim3 dimGrid(blocks_each, blocks_each), dimBlock(threads_each, threads_each);

    int leastPriority = 0;
    int greatestPriority = leastPriority;
    CUDA_RT_CALL(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

#pragma omp parallel num_threads(num_devices)
    {
        int dev_id = omp_get_thread_num();

        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaFree(nullptr));

        // For debugging locally
        if (num_devices > 1) {
            int canAccessPeer = 0;
            const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);

            CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, top));
            if (canAccessPeer) {
                CUDA_RT_CALL(cudaDeviceEnablePeerAccess(top, 0));
            } else {
                std::cerr << "P2P access required from " << dev_id << " to " << top << std::endl;
                std::exit(1);
            }
        }

        CUDA_RT_CALL(cudaMalloc(&a, nx * ny * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(&a_new, nx * ny * sizeof(real)));

        CUDA_RT_CALL(cudaMemset(a, 0, nx * ny * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(a_new, 0, nx * ny * sizeof(real)));

        // Set diriclet boundary conditions on left and right boarder
        initialize_boundaries<<<ny / 128 + 1, 128>>>(a, a_new, PI, nx, ny);
        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());

        if (!csv)
            printf(
                "Jacobi relaxation: %d iterations on %d x %d mesh with norm check "
                "every %d iterations\n",
                iter_max, ny, nx, nccheck);

        PUSH_RANGE("Jacobi solve", 0)

        int* flag;
        CUDA_RT_CALL(cudaMalloc(&flag, 1 * sizeof(int)));
        CUDA_RT_CALL(cudaMemset(flag, 0, 1 * sizeof(int)));

        void* kernelArgs[] = {(void*)&a_new, (void*)&a,        (void*)&iy_start, (void*)&iy_end,
                              (void*)&nx,    (void*)&iter_max, (void*)&flag};

        cudaStream_t inner_domain_stream;
        cudaStream_t boundary_sync_stream;

        // Creating streams with priority
        CUDA_RT_CALL(cudaStreamCreateWithPriority(&inner_domain_stream, cudaStreamNonBlocking,
                                                  greatestPriority));
        CUDA_RT_CALL(cudaStreamCreateWithPriority(&boundary_sync_stream, cudaStreamNonBlocking,
                                                  leastPriority));

        CUDA_RT_CALL(cudaSetDevice(dev_id));

        // Inner domain
        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void*)jacobi_kernel, dimGrid, dimBlock,
                                                 kernelArgs, 0, inner_domain_stream));

        // Boundary
        boundary_sync_kernel<<<1, 1, 0, boundary_sync_stream>>>(a, flag);

        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());
    }
}
