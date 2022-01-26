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
#include "../include/multi-gpu-peer.cuh"
#include "../include/single-gpu-naive.cuh"

namespace cg = cooperative_groups;

constexpr real ZERO_TWENTY_FIVE { 0.25 };

namespace MultiGPUPeer {
    __global__ void initialize_boundaries(real* __restrict__ const a_new, real* __restrict__ const a,
                                          const real pi, const int offset, const int nx,
                                          const int my_ny, const int ny) {
        for (unsigned int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < my_ny;
             iy += blockDim.x * gridDim.x) {
            const real y0 = sin(2.0 * pi * (offset + iy) / (ny - 1));
            a[iy * nx + 0] = y0;
            a[iy * nx + (nx - 1)] = y0;
            a_new[iy * nx + 0] = y0;
            a_new[iy * nx + (nx - 1)] = y0;
        }
    }
}

__global__ void jacobi_kernel(real* a_new, real* a, const int iy_start, const int iy_end,
                              const int nx, real* a_new_top, const int top_iy, real* a_new_bottom,
                              const int bottom_iy, const int iter_max, int* debug_flag) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;

    int iter = 0;

    while (iter < iter_max) {
        if (iy > iy_start && iy < iy_end - 1 && ix < (nx - 1)) {
            const real new_val = ZERO_TWENTY_FIVE * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                                     a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
            a_new[iy * nx + ix] = new_val;
        }

        real* temp_pointer_first = a_new;
        a_new = a;
        a = temp_pointer_first;

        iter++;

        grid.sync();
    }

    *debug_flag = 1;
}

__global__ void boundary_sync_kernel(const real* a, const int iy_start, const int iy_end,
                                     const int nx, real* a_new_top, const int top_iy, real* a_new_bottom,
                                     const int bottom_iy, const int iter,
                                     const volatile int* local_is_top_neighbor_done_writing_to_me,
                                     const volatile int* local_is_bottom_neighbor_done_writing_to_me,
                                     volatile int* remote_am_done_writing_to_top_neighbor,
                                     volatile int* remote_am_done_writing_to_bottom_neighbor,
                                     [[maybe_unused]] volatile const int* debug_flag) {
    unsigned int iy = threadIdx.y + iy_start;
    unsigned int ix = threadIdx.x + 1;
    unsigned int col = iy * blockDim.x + ix;

    //    while (!*debug_flag) {
    //    }

    //    printf("%d %d\n", col, nx);

    if (col < nx) {
        // Wait until top GPU puts its bottom row as my top halo
        while (local_is_top_neighbor_done_writing_to_me[0] != iter) {
        }

        const real first_row_val =
            ZERO_TWENTY_FIVE * (a[iy_start * nx + col + 1] + a[iy_start * nx + col - 1] +
                                a[(iy_start + 1) * nx + col] + a[(iy_start - 1) * nx + col]);

        while (local_is_bottom_neighbor_done_writing_to_me[0] != 1) {
        }

        const real last_row_val =
            ZERO_TWENTY_FIVE * (a[(iy_end - 1) * nx + col + 1] + a[(iy_end - 1) * nx + col - 1] +
                                a[(iy_end - 2) * nx + col] + a[(iy_end)*nx + col]);

        // Communication
        a_new_top[top_iy * nx + col] = first_row_val;
        a_new_bottom[bottom_iy * nx + col] = last_row_val;
    }

    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        remote_am_done_writing_to_top_neighbor[1] = 1;
        remote_am_done_writing_to_bottom_neighbor[1] = 1;
    }

    printf("ok\n");
}

constexpr int THREADS_PER_BLOCK = 1024;

int MultiGPUPeer::init(int argc, char** argv) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nccheck = get_argval<int>(argv, argv + argc, "-nccheck", 1);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);
    //    const bool csv = get_arg(argv, argv + argc, "-csv");

    if (nccheck != 1) {
        fprintf(stderr, "Only nccheck = 1 is supported\n");
        return -1;
    }

    printf(
        "Jacobi relaxation: %d iterations on %d x %d mesh with norm check "
        "every %d iterations\n",
        iter_max, ny, nx, nccheck);

    real* a;
    real* a_new[MAX_NUM_DEVICES];

    int iy_start = 1;
    int iy_end[MAX_NUM_DEVICES];

    int* is_top_done_computing_flags[MAX_NUM_DEVICES];
    int* is_bottom_done_computing_flags[MAX_NUM_DEVICES];

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

    // Chunks for each GPU
    int chunk_size;
    int chunk_size_low = (ny - 2) / num_devices;
    int chunk_size_high = chunk_size_low + 1;

    int num_ranks_low = num_devices * chunk_size_low + num_devices - (ny - 2);

#pragma omp parallel num_threads(num_devices)
    {
        const int dev_id = omp_get_thread_num();

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

#pragma omp barrier
        if (dev_id < num_ranks_low) {
            chunk_size = chunk_size_low;
        } else {
            chunk_size = chunk_size_high;
        }

        const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);
        const int bottom = (dev_id + 1) % num_devices;

#pragma omp barrier

        CUDA_RT_CALL(cudaMalloc(&a, nx * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(&a_new[dev_id], nx * (chunk_size + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMemset(a, 0, nx * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(a_new[dev_id], 0, nx * (chunk_size + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMalloc(&is_top_done_computing_flags[dev_id], 2 * sizeof(int)));
        CUDA_RT_CALL(cudaMalloc(&is_bottom_done_computing_flags[dev_id], 2 * sizeof(int)));

        CUDA_RT_CALL(cudaMemset(is_top_done_computing_flags[dev_id], 0, sizeof(int)));
        CUDA_RT_CALL(cudaMemset(is_bottom_done_computing_flags[dev_id], 0, sizeof(int)));

        // Calculate local domain boundaries
        int iy_start_global;  // My start index in the global array
        if (dev_id < num_ranks_low) {
            iy_start_global = dev_id * chunk_size_low + 1;
        } else {
            iy_start_global =
                num_ranks_low * chunk_size_low + (dev_id - num_ranks_low) * chunk_size_high + 1;
        }
        int iy_end_global = iy_start_global + chunk_size - 1;  // My last index in the global array

        iy_end[dev_id] = (iy_end_global - iy_start_global + 1) + iy_start;
        int iy_start_bottom = 0;

        // Set dirichlet boundary conditions on left and right border
        MultiGPUPeer::initialize_boundaries<<<(ny / num_devices) / 128 + 1, 128>>>(
            a, a_new[dev_id], PI, iy_start_global - 1, nx, (chunk_size + 2), ny);

        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());

        int* flag;
        CUDA_RT_CALL(cudaMalloc(&flag, 1 * sizeof(int)));
        CUDA_RT_CALL(cudaMemset(flag, 0, 1 * sizeof(int)));

        void* kernelArgs[] = {
            (void*)&a_new[dev_id],
            (void*)&a,
            (void*)&iy_start,
            (void*)&iy_end[dev_id],
            (void*)&nx,
            (void*)&a_new[top],
            (void*)&iy_end[top],
            (void*)&a_new[bottom],
            (void*)&iy_start_bottom,
            (void*)&iter_max,
            (void*)&flag
        };

        cudaStream_t inner_domain_stream;
        cudaStream_t boundary_sync_stream;

        // Creating streams with priority
        CUDA_RT_CALL(cudaStreamCreateWithPriority(&inner_domain_stream, cudaStreamNonBlocking,
                                                  greatestPriority));
        CUDA_RT_CALL(cudaStreamCreateWithPriority(&boundary_sync_stream, cudaStreamNonBlocking,
                                                  leastPriority));

        CUDA_RT_CALL(cudaSetDevice(dev_id));

#pragma omp barrier

        // Inner domain
        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void*)jacobi_kernel, dimGrid, dimBlock,
                                                 kernelArgs, 0, inner_domain_stream));

        for (int iter = 0; iter < iter_max; iter++) {
            // Boundary
            boundary_sync_kernel<<<1, dimBlock, 0, boundary_sync_stream>>>(
                a, iy_start, iy_end[dev_id], nx, a_new[top], iy_end[top], a_new[bottom],
                iter, iy_start_bottom, is_top_done_computing_flags[dev_id],
                is_bottom_done_computing_flags[dev_id], is_bottom_done_computing_flags[top],
                is_top_done_computing_flags[bottom], flag);

            CUDA_RT_CALL(cudaGetLastError());
            CUDA_RT_CALL(cudaDeviceSynchronize());
        }

        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());
    }

    return 0;
};
