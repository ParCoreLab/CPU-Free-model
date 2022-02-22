/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */
#include <cmath>
#include <cstdio>
#include <iostream>

#include <omp.h>

#include <cooperative_groups.h>

#include "../../include/common.h"
#include "../../include/single-stream/multi-threaded-two-block-comm.cuh"

namespace cg = cooperative_groups;

namespace SSMultiThreadedTwoBlockComm {
    __global__ void __launch_bounds__(1024, 1)
    jacobi_kernel(real *a_new, real *a, const int iy_start, const int iy_end, const int nx,
                  const int tile_size, const int num_tiles_x, const int num_tiles_y,
                  real *a_new_top, real *a_top, const int top_iy, real *a_new_bottom,
                  real *a_bottom, const int bottom_iy, const int iter_max,
                  volatile int *local_is_top_neighbor_done_writing_to_me,
                  volatile int *local_is_bottom_neighbor_done_writing_to_me,
                  volatile int *remote_am_done_writing_to_top_neighbor,
                  volatile int *remote_am_done_writing_to_bottom_neighbor) {
        cg::thread_block cta = cg::this_thread_block();
        cg::grid_group grid = cg::this_grid();

        unsigned int grid_dim_x = (tile_size + blockDim.x - 1) / blockDim.x;
        unsigned int block_idx_y = blockIdx.x / grid_dim_x;
        unsigned int block_idx_x = blockIdx.x % grid_dim_x;

        unsigned int base_iy = block_idx_y * blockDim.y + threadIdx.y + iy_start;
        unsigned int base_ix = block_idx_x * blockDim.x + threadIdx.x + 1;

        int num_flags = 2 * num_tiles_x;

        int iter = 0;

        int cur_iter_mod = 0;
        int next_iter_mod = 1;
        int temp_iter_mod = 0;

        int tile_start_ny = 0;
        int tile_end_ny = 0;
        int tile_start_nx = 0;
        int tile_end_nx = 0;

        int cur_iter_tile_flag_idx;
        int next_iter_tile_flag_idx;

        while (iter < iter_max) {
            for (int tile_idx_y = 0; tile_idx_y < num_tiles_y; tile_idx_y++) {
                unsigned int iy = base_iy + tile_idx_y * tile_size - iy_start * (tile_idx_y != 0);

                tile_start_ny = tile_idx_y * tile_size + iy_start * (tile_idx_y == 0);
                tile_end_ny =
                        (tile_idx_y == num_tiles_y - 1) ? iy_end - 1 : (tile_idx_y + 1) * tile_size;

                for (int tile_idx_x = 0; tile_idx_x < num_tiles_x; tile_idx_x++) {
                    unsigned int ix = base_ix + tile_idx_x * tile_size - (tile_idx_x != 0);

                    tile_start_nx = tile_idx_x * tile_size + (tile_idx_x == 0);
                    tile_end_nx = (tile_idx_x + 1) * tile_size - (tile_idx_x == num_tiles_x - 1);

                    tile_end_nx = min(tile_end_nx, nx - 1);

                    //    One thread block does communication (and a bit of computation)
                    if (blockIdx.x == gridDim.x - 1) {
                        if (tile_idx_y == 0) {
                            unsigned int col = threadIdx.y * blockDim.x + threadIdx.x + tile_start_nx;

                            cur_iter_tile_flag_idx = tile_idx_x + cur_iter_mod * num_flags;
                            next_iter_tile_flag_idx =
                                    (num_tiles_x + tile_idx_x) % num_tiles_x + next_iter_mod * num_flags;

                            if (col <= (tile_end_nx - 1)) {
                                // Wait until top GPU puts its bottom row as my top halo
                                while (
                                        local_is_top_neighbor_done_writing_to_me[cur_iter_tile_flag_idx] !=
                                        iter) {
                                }

                                const real first_row_val =
                                        0.25 *
                                        (a[iy_start * nx + col + 1] + a[iy_start * nx + col - 1] +
                                         a[(iy_start + 1) * nx + col] + a[(iy_start - 1) * nx + col]);
                                a_new[iy_start * nx + col] = first_row_val;

                                // Communication
                                a_new_top[top_iy * nx + col] = first_row_val;
                            }

                            cg::sync(cta);

                            if (threadIdx.x == 0 && threadIdx.y == 0) {
                                remote_am_done_writing_to_top_neighbor[next_iter_tile_flag_idx] =
                                        iter + 1;
                            }
                        }
                    } else if (blockIdx.x == gridDim.x - 2) {
                        if (tile_idx_y == num_tiles_y - 1) {
                            unsigned int col = threadIdx.y * blockDim.x + threadIdx.x + tile_start_nx;

                            cur_iter_tile_flag_idx =
                                    (num_tiles_x + tile_idx_x) % num_tiles_x + cur_iter_mod * num_flags;
                            next_iter_tile_flag_idx = tile_idx_x + next_iter_mod * num_flags;

                            if (col <= (tile_end_nx - 1)) {
                                while (local_is_bottom_neighbor_done_writing_to_me
                                       [cur_iter_tile_flag_idx] != iter) {
                                }

                                const real last_row_val =
                                        0.25 *
                                        (a[(iy_end - 1) * nx + col + 1] + a[(iy_end - 1) * nx + col - 1] +
                                         a[(iy_end - 2) * nx + col] + a[iy_end * nx + col]);
                                a_new[(iy_end - 1) * nx + col] = last_row_val;

                                // Communication
                                a_new_bottom[bottom_iy * nx + col] = last_row_val;
                            }

                            cg::sync(cta);

                            if (threadIdx.x == 0 && threadIdx.y == 0) {
                                remote_am_done_writing_to_bottom_neighbor[next_iter_tile_flag_idx] =
                                        iter + 1;
                            }
                        }
                    } else if (iy >= tile_start_ny && iy <= (tile_end_ny - 1) && ix >= tile_start_nx &&
                               ix <= (tile_end_nx - 1)) {
                        const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                                     a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
                        a_new[iy * nx + ix] = new_val;
                    }
                }
            }

            real *temp_pointer_first = a_new;
            a_new = a;
            a = temp_pointer_first;

            real *temp_pointer_second = a_new_top;
            a_new_top = a_top;
            a_top = temp_pointer_second;

            real *temp_pointer_third = a_new_bottom;
            a_new_bottom = a_bottom;
            a_bottom = temp_pointer_third;

            iter++;

            temp_iter_mod = cur_iter_mod;
            cur_iter_mod = next_iter_mod;
            next_iter_mod = temp_iter_mod;

            cg::sync(grid);
        }
    }
}  // namespace SSMultiThreadedTwoBlockComm

int SSMultiThreadedTwoBlockComm::init(int argc, char *argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);
    const bool compare_to_single_gpu = get_arg(argv, argv + argc, "-compare");

    real *a[MAX_NUM_DEVICES];
    real *a_new[MAX_NUM_DEVICES];
    int iy_end[MAX_NUM_DEVICES];

    real *a_ref_h;
    real *a_h;

    double runtime_serial_non_persistent = 0.0;

    int *is_top_done_computing_flags[MAX_NUM_DEVICES];
    int *is_bottom_done_computing_flags[MAX_NUM_DEVICES];

    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));
    //    real l2_norm = 1.0;

#pragma omp parallel num_threads(num_devices)
    {
        //        real* l2_norm_d;
        //        real* l2_norm_h;

        int dev_id = omp_get_thread_num();

        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaFree(nullptr));

        if (compare_to_single_gpu && 0 == dev_id) {
            CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * sizeof(real)));
            CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * sizeof(real)));

            // Passing 0 for nccheck for now
            runtime_serial_non_persistent = single_gpu(nx, ny, iter_max, a_ref_h, 0, true);
        }

#pragma omp barrier

        int chunk_size;
        int chunk_size_low = (ny - 2) / num_devices;
        int chunk_size_high = chunk_size_low + 1;

        // For now, I am assuming the height is exactly divisible by the number of devices
        int height_per_gpu = ny / num_devices;

        // A tile will be TILE_SIZE in both dimensions
        int num_tiles_x = nx / TILE_SIZE + (nx % TILE_SIZE != 0);
        int num_tiles_y = height_per_gpu / TILE_SIZE + (height_per_gpu % TILE_SIZE != 0);
        int num_flags = 4 * num_tiles_x;

        int num_ranks_low = num_devices * chunk_size_low + num_devices - (ny - 2);
        if (dev_id < num_ranks_low)
            chunk_size = chunk_size_low;
        else
            chunk_size = chunk_size_high;

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

#pragma omp barrier

        CUDA_RT_CALL(cudaMalloc(a + dev_id, nx * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(a_new + dev_id, nx * (chunk_size + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMemset(a[dev_id], 0, nx * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(a_new[dev_id], 0, nx * (chunk_size + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMalloc(is_top_done_computing_flags + dev_id, num_flags * sizeof(int)));
        CUDA_RT_CALL(cudaMalloc(is_bottom_done_computing_flags + dev_id, num_flags * sizeof(int)));

        CUDA_RT_CALL(cudaMemset(is_top_done_computing_flags[dev_id], 0, num_flags * sizeof(int)));
        CUDA_RT_CALL(
                cudaMemset(is_bottom_done_computing_flags[dev_id], 0, num_flags * sizeof(int)));

        // Calculate local domain boundaries
        int iy_start_global;  // My start index in the global array
        if (dev_id < num_ranks_low) {
            iy_start_global = dev_id * chunk_size_low + 1;
        } else {
            iy_start_global =
                    num_ranks_low * chunk_size_low + (dev_id - num_ranks_low) * chunk_size_high + 1;
        }
        int iy_end_global = iy_start_global + chunk_size - 1;  // My last index in the global array

        int iy_start = 1;
        iy_end[dev_id] = (iy_end_global - iy_start_global + 1) + iy_start;
        int iy_start_bottom = 0;

        // Set diriclet boundary conditions on left and right border
        initialize_boundaries<<<(ny / num_devices) / 128 + 1, 128>>>(
                a[dev_id], a_new[dev_id], PI, iy_start_global - 1, nx, chunk_size + 2, ny);
        CUDA_RT_CALL(cudaGetLastError());

        CUDA_RT_CALL(cudaDeviceSynchronize());

        constexpr int dim_block_x = 32;
        constexpr int dim_block_y = 32;

        cudaDeviceProp deviceProp{};
        CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, dev_id));
        int numSms = deviceProp.multiProcessorCount;

        dim3 dim_grid(numSms, 1, 1);
        dim3 dim_block(dim_block_x, dim_block_y);

        void *kernelArgs[] = {(void *) &a_new[dev_id],
                              (void *) &a[dev_id],
                              (void *) &iy_start,
                              (void *) &iy_end[dev_id],
                              (void *) &nx,
                              (void *) &TILE_SIZE,
                              (void *) &num_tiles_x,
                              (void *) &num_tiles_y,
                              (void *) &a_new[top],
                              (void *) &a[top],
                              (void *) &iy_end[top],
                              (void *) &a_new[bottom],
                              (void *) &a[bottom],
                              (void *) &iy_start_bottom,
                              (void *) &iter_max,
                              (void *) &is_top_done_computing_flags[dev_id],
                              (void *) &is_bottom_done_computing_flags[dev_id],
                              (void *) &is_bottom_done_computing_flags[top],
                              (void *) &is_top_done_computing_flags[bottom]};

#pragma omp barrier
        double start = omp_get_wtime();

        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *) SSMultiThreadedTwoBlockComm::jacobi_kernel,
                                                 dim_grid, dim_block, kernelArgs, 0, nullptr));

        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());

#pragma omp barrier
        double stop = omp_get_wtime();

        if (compare_to_single_gpu) {
            CUDA_RT_CALL(
                    cudaMemcpy(a_h + iy_start_global * nx, a[dev_id] + nx,
                               std::min((ny - iy_start_global) * nx, chunk_size * nx) * sizeof(real),
                               cudaMemcpyDeviceToHost));
        }

#pragma omp barrier

#pragma omp master
        {
            report_results(ny, nx, a_ref_h, a_h, num_devices, runtime_serial_non_persistent, start,
                           stop, compare_to_single_gpu);
        }

        CUDA_RT_CALL(cudaFree(a_new[dev_id]));
        CUDA_RT_CALL(cudaFree(a[dev_id]));

        if (compare_to_single_gpu && 0 == dev_id) {
            CUDA_RT_CALL(cudaFreeHost(a_h));
            CUDA_RT_CALL(cudaFreeHost(a_ref_h));
        }
    }

    return 0;
}