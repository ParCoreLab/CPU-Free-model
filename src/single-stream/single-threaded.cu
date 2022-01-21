/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */
#include <cmath>
#include <cstdio>
#include <iostream>

#include <cooperative_groups.h>

#include "../../include/common.h"
#include "../../include/single-stream/single-threaded.cuh"

namespace cg = cooperative_groups;

constexpr int MAX_NUM_DEVICES = 32;
typedef float real;

const real PI = 2.0 * std::asin(1.0);

namespace SSSingleThreaded {
    __global__ void initialize_boundaries(real* __restrict__ const a_new, real* __restrict__ const a,
                                          const real pi, const int offset, const int nx,
                                          const int my_ny, const int ny) {
        for (int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < my_ny; iy += blockDim.x * gridDim.x) {
            const real y0 = sin(2.0 * pi * (offset + iy) / (ny - 1));
            a[iy * nx + 0] = y0;
            a[iy * nx + (nx - 1)] = y0;
            a_new[iy * nx + 0] = y0;
            a_new[iy * nx + (nx - 1)] = y0;
        }
    }
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void jacobi_kernel(real* __restrict__ a_new, const real* __restrict__ a,
                              const int iy_start, const int iy_end, const int nx,
                              real* __restrict__ const a_new_top, const int top_iy,
                              real* __restrict__ const a_new_bottom, const int bottom_iy,
                              const int iter_max, int* is_top_neigbor_done,
                              int* is_bottom_neigbor_done, int* notify_top_neighbor,
                              int* notify_bottom_neighbor) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;

    real local_l2_norm = 0.0;
    int i = 0;

    while (i < iter_max) {
        //    One thread block does communication (and a bit of computation)
        if (blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y - 1) {
            int iy = threadIdx.y + iy_start;
            int ix = threadIdx.x + 1;
            int col = iy * blockDim.x + ix;

            if (col < nx) {
                // Wait until top GPU puts its bottom row as my top halo
                while (!*is_top_neigbor_done) {
                }

                const real first_row_val =
                    0.25 * (a[iy_start * nx + col + 1] + a[iy_start * nx + col - 1] +
                            a[(iy_start + 1) * nx + col] + a[(iy_start - 1) * nx + col]);
                a_new_top[top_iy * nx + col] = first_row_val;

                // Wait until bottom GPU puts its top row as my bottom halo

                while (!*is_bottom_neigbor_done) {
                }

                const real last_row_val =
                    0.25 * (a[(iy_end - 1) * nx + col + 1] + a[(iy_end - 1) * nx + col - 1] +
                            a[(iy_end - 2) * nx + col] + a[(iy_end)*nx + col]);
                a_new_bottom[bottom_iy * nx + col] = last_row_val;
            }

            cta.sync();

            if (threadIdx.x == 0 && threadIdx.y == 0) {
                *notify_bottom_neighbor = 1;
                *notify_top_neighbor = 1;
            }
        } else if (iy > iy_start && iy < iy_end - 1 && ix < (nx - 1)) {
            const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                         a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
            a_new[iy * nx + ix] = new_val;

            real residue = new_val - a[iy * nx + ix];
            local_l2_norm = residue * residue;
        }

        real* temp_pointer = a_new;
        a = a_new;
        a_new = temp_pointer;

        i++;

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            *notify_top_neighbor = 0;
            *notify_bottom_neighbor = 0;
        }

        grid.sync();
    }
}

int SSSingleThreaded::init(int argc, char* argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);

    real* a_new[MAX_NUM_DEVICES];
    real* a[MAX_NUM_DEVICES];

    int iy_start[MAX_NUM_DEVICES];
    int iy_end[MAX_NUM_DEVICES];

    int chunk_size[MAX_NUM_DEVICES];

    int* is_top_done_computing_flags[MAX_NUM_DEVICES];
    int* is_bottom_done_computing_flags[MAX_NUM_DEVICES];

    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

    for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaFree(0));

        int chunk_size_low = (ny - 2) / num_devices;
        int chunk_size_high = chunk_size_low + 1;

        int num_ranks_low = num_devices * chunk_size_low + num_devices - (ny - 2);
        if (dev_id < num_ranks_low)
            chunk_size[dev_id] = chunk_size_low;
        else
            chunk_size[dev_id] = chunk_size_high;

        const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);
        const int bottom = (dev_id + 1) % num_devices;

        if (top != dev_id) {
            int canAccessPeer = 0;
            CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, top));
            if (canAccessPeer) {
                CUDA_RT_CALL(cudaDeviceEnablePeerAccess(top, 0));
            } else {
                std::cerr << "P2P access required from " << dev_id << " to " << top << std::endl;
            }
            if (top != bottom) {
                canAccessPeer = 0;
                CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, bottom));
                if (canAccessPeer) {
                    CUDA_RT_CALL(cudaDeviceEnablePeerAccess(bottom, 0));
                } else {
                    std::cerr << "P2P access required from " << dev_id << " to " << bottom
                              << std::endl;
                }
            }
        }

        CUDA_RT_CALL(cudaMalloc(a + dev_id, nx * (chunk_size[dev_id] + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(a_new + dev_id, nx * (chunk_size[dev_id] + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMemset(a[dev_id], 0, nx * (chunk_size[dev_id] + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(a_new[dev_id], 0, nx * (chunk_size[dev_id] + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMalloc(is_top_done_computing_flags + dev_id, 1 * sizeof(int)));
        CUDA_RT_CALL(cudaMalloc(is_bottom_done_computing_flags + dev_id, 1 * sizeof(int)));

        CUDA_RT_CALL(cudaMemset(is_top_done_computing_flags[dev_id], 0, 1 * sizeof(int)));
        CUDA_RT_CALL(cudaMemset(is_bottom_done_computing_flags[dev_id], 0, 1 * sizeof(int)));

        // Calculate local domain boundaries
        int iy_start_global;  // My start index in the global array
        if (dev_id < num_ranks_low) {
            iy_start_global = dev_id * chunk_size_low + 1;
        } else {
            iy_start_global =
                num_ranks_low * chunk_size_low + (dev_id - num_ranks_low) * chunk_size_high + 1;
        }

        iy_start[dev_id] = 1;
        iy_end[dev_id] = iy_start[dev_id] + chunk_size[dev_id];

        // Set diriclet boundary conditions on left and right border
        SSSingleThreaded::initialize_boundaries<<<(ny / num_devices) / 128 + 1, 128>>>(
            a[dev_id], a_new[dev_id], PI, iy_start_global - 1, nx, (chunk_size[dev_id] + 2), ny);
        CUDA_RT_CALL(cudaGetLastError());

        CUDA_RT_CALL(cudaDeviceSynchronize());

        constexpr int dim_block_x = 16;
        constexpr int dim_block_y = 16;

        int* notify_top = dev_id > 0 ? is_bottom_done_computing_flags[dev_id - 1]
                                     : is_bottom_done_computing_flags[num_devices - 1];
        int* notify_bottom = dev_id < num_devices - 1 ? is_top_done_computing_flags[dev_id + 1]
                                                      : is_top_done_computing_flags[0];

        void* kernelArgs[] = {
            (void*)&a_new[dev_id],
            (void*)&a[dev_id],
            (void*)&iy_start,
            (void*)&iy_end[dev_id],
            (void*)&nx,
            (void*)&a_new[top],
            (void*)&iy_end[top],
            (void*)&a_new[bottom],
//            (void*)&iy_start_bottom,
            (void*)&iter_max,
            (void*)&is_top_done_computing_flags[dev_id],
            (void*)&is_bottom_done_computing_flags[dev_id],
            (void*)&notify_top,
            (void*)&notify_bottom,
        };

        cudaDeviceProp deviceProp{};
        CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, dev_id));
        int numSms = deviceProp.multiProcessorCount;

        constexpr int THREADS_PER_BLOCK = 256;

        int sMemSize = sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1);
        int numBlocksPerSm = 0;
        int numThreads = THREADS_PER_BLOCK;

        CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocksPerSm, jacobi_kernel<dim_block_x, dim_block_y>, numThreads, 0));

        int blocks_each = (int)sqrt(numSms * numBlocksPerSm);
        int threads_each = (int)sqrt(THREADS_PER_BLOCK);
        dim3 dimGrid(blocks_each, blocks_each), dimBlock(threads_each, threads_each);

        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void*)jacobi_kernel<dim_block_x, dim_block_y>,
                                                 dimGrid, dimBlock, kernelArgs, 0, nullptr));
    }

    for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
        CUDA_RT_CALL(cudaGetLastError());

        CUDA_RT_CALL(cudaDeviceSynchronize());
    }
}