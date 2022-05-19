/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */
#include <cmath>
#include <cstdio>
#include <iostream>

#include <omp.h>

#include <cooperative_groups.h>

#include "../../include/common.h"
#include "../../include/multi-stream/multi-gpu-peer-tiling.cuh"

namespace cg = cooperative_groups;

namespace MultiGPUPeerTiling {
__global__ void __launch_bounds__(1024, 1) jacobi_kernel(
    real *a_new, real *a, const int iy_start, const int iy_end, const int nx, const int tile_size,
    const int num_tiles_x, const int num_tiles_y, const int top_iy, const int bottom_iy,
    const int iter_max, real *local_halo_buffer_for_top_neighbor,
    real *local_halo_buffer_for_bottom_neighbor, real *remote_my_halo_buffer_on_top_neighbor,
    real *remote_my_halo_buffer_on_bottom_neighbor,
    volatile int *local_is_top_neighbor_done_writing_to_me,
    volatile int *local_is_bottom_neighbor_done_writing_to_me,
    volatile int *remote_am_done_writing_to_top_neighbor,
    volatile int *remote_am_done_writing_to_bottom_neighbor, volatile int *iteration_done) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    int grid_dim_x = (tile_size + blockDim.x - 1) / blockDim.x;
    int block_idx_y = blockIdx.x / grid_dim_x;
    int block_idx_x = blockIdx.x % grid_dim_x;

    int base_iy = block_idx_y * blockDim.y + threadIdx.y;
    int base_ix = block_idx_x * blockDim.x + threadIdx.x;

    int num_flags = 2 * num_tiles_x;

    int iter = 0;

    int cur_iter_mod = 0;
    int next_iter_mod = 1;
    int temp_iter_mod = 0;

    int cur_iter_tile_flag_idx;
    int next_iter_tile_flag_idx;

    int tile_idx_x;
    int tile_idx_y;

    int tile_start_ny;
    int tile_end_ny;
    int tile_start_nx;
    int tile_end_nx;

    int iy;
    int ix;

    while (iter < iter_max) {
        for (tile_idx_y = 0; tile_idx_y < num_tiles_y; tile_idx_y++) {
            for (tile_idx_x = 0; tile_idx_x < num_tiles_x; tile_idx_x++) {
                tile_start_ny = (tile_idx_y == 0) ? iy_start + 1 : tile_idx_y * tile_size;
                tile_end_ny =
                    (tile_idx_y == (num_tiles_y - 1)) ? iy_end - 1 : (tile_idx_y + 1) * tile_size;

                tile_start_nx = (tile_idx_x == 0) ? 1 : tile_idx_x * tile_size;
                tile_end_nx =
                    (tile_idx_x == (num_tiles_x - 1)) ? nx - 1 : (tile_idx_x + 1) * tile_size;

                iy = base_iy + tile_start_ny;
                ix = base_ix + tile_start_nx;

                if (iy < tile_end_ny && ix < tile_end_nx) {
                    const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                                 a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
                    a_new[iy * nx + ix] = new_val;
                }
            }
        }

        real *temp_pointer_first = a_new;
        a_new = a;
        a = temp_pointer_first;

        iter++;

        temp_iter_mod = cur_iter_mod;
        cur_iter_mod = next_iter_mod;
        next_iter_mod = temp_iter_mod;

        cg::sync(grid);

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            while (iteration_done[0] != iter) {
            }
            iteration_done[1] = iter;
        }

        cg::sync(grid);
    }
}

__global__ void __launch_bounds__(1024, 1) boundary_sync_kernel(
    real *a_new, real *a, const int iy_start, const int iy_end, const int nx, const int tile_size,
    const int num_tiles_x, const int num_tiles_y, const int top_iy, const int bottom_iy,
    const int iter_max, real *local_halo_buffer_for_top_neighbor,
    real *local_halo_buffer_for_bottom_neighbor, real *remote_my_halo_buffer_on_top_neighbor,
    real *remote_my_halo_buffer_on_bottom_neighbor,
    volatile int *local_is_top_neighbor_done_writing_to_me,
    volatile int *local_is_bottom_neighbor_done_writing_to_me,
    volatile int *remote_am_done_writing_to_top_neighbor,
    volatile int *remote_am_done_writing_to_bottom_neighbor, volatile int *iteration_done) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    int grid_dim_x = (tile_size + blockDim.x - 1) / blockDim.x;
    int block_idx_y = blockIdx.x / grid_dim_x;
    int block_idx_x = blockIdx.x % grid_dim_x;

    int base_iy = block_idx_y * blockDim.y + threadIdx.y;
    int base_ix = block_idx_x * blockDim.x + threadIdx.x;

    int num_flags = 2 * num_tiles_x;

    int iter = 0;

    int cur_iter_mod = 0;
    int next_iter_mod = 1;
    int temp_iter_mod = 0;

    int cur_iter_tile_flag_idx;
    int next_iter_tile_flag_idx;

    int tile_idx_x;
    int tile_idx_y;

    int tile_start_ny;
    int tile_end_ny;
    int tile_start_nx;
    int tile_end_nx;

    int iy;
    int ix;

    while (iter < iter_max) {
        while (iteration_done[1] != iter) {}

        for (tile_idx_y = 0; tile_idx_y < num_tiles_y; tile_idx_y++) {
            for (tile_idx_x = 0; tile_idx_x < num_tiles_x; tile_idx_x++) {
                tile_start_ny = (tile_idx_y == 0) ? iy_start + 1 : tile_idx_y * tile_size;
                tile_end_ny =
                    (tile_idx_y == (num_tiles_y - 1)) ? iy_end - 1 : (tile_idx_y + 1) * tile_size;

                tile_start_nx = (tile_idx_x == 0) ? 1 : tile_idx_x * tile_size;
                tile_end_nx =
                    (tile_idx_x == (num_tiles_x - 1)) ? nx - 1 : (tile_idx_x + 1) * tile_size;

                iy = base_iy + tile_start_ny;
                ix = base_ix + tile_start_nx;

                if (blockIdx.x == gridDim.x - 1) {
                    int col = threadIdx.y * blockDim.x + threadIdx.x + tile_start_nx;

                    cur_iter_tile_flag_idx = tile_idx_x + cur_iter_mod * num_flags;
                    next_iter_tile_flag_idx =
                        (num_tiles_x + tile_idx_x) + next_iter_mod * num_flags;

                    if (tile_idx_y == 0) {
                        if (cta.thread_rank() == 0) {
                            while (
                                local_is_top_neighbor_done_writing_to_me[cur_iter_tile_flag_idx] !=
                                iter) {
                            }
                        }

                        cg::sync(cta);

                        if (col < tile_end_nx) {
                            const real first_row_val =
                                0.25 *
                                (a[iy_start * nx + col + 1] + a[iy_start * nx + col - 1] +
                                 a[(iy_start + 1) * nx + col] +
                                 remote_my_halo_buffer_on_top_neighbor[nx * cur_iter_mod + col]);

                            a_new[iy_start * nx + col] = first_row_val;
                            local_halo_buffer_for_top_neighbor[nx * next_iter_mod + col] =
                                first_row_val;
                        }

                        cg::sync(cta);

                        if (cta.thread_rank() == 0) {
                            remote_am_done_writing_to_top_neighbor[next_iter_tile_flag_idx] =
                                iter + 1;
                        }
                    }
                } else if (blockIdx.x == gridDim.x - 2) {
                    int col = threadIdx.y * blockDim.x + threadIdx.x + tile_start_nx;

                    cur_iter_tile_flag_idx = (num_tiles_x + tile_idx_x) + cur_iter_mod * num_flags;
                    next_iter_tile_flag_idx = tile_idx_x + next_iter_mod * num_flags;

                    if (tile_idx_y == (num_tiles_y - 1)) {
                        if (cta.thread_rank() == 0) {
                            while (local_is_bottom_neighbor_done_writing_to_me
                                       [cur_iter_tile_flag_idx] != iter) {
                            }
                        }

                        cg::sync(cta);

                        if (col < tile_end_nx) {
                            const real last_row_val =
                                0.25 *
                                (a[(iy_end - 1) * nx + col + 1] + a[(iy_end - 1) * nx + col - 1] +
                                 remote_my_halo_buffer_on_bottom_neighbor[nx * cur_iter_mod + col] +
                                 a[(iy_end - 2) * nx + col]);

                            a_new[(iy_end - 1) * nx + col] = last_row_val;
                            local_halo_buffer_for_bottom_neighbor[nx * next_iter_mod + col] =
                                last_row_val;
                        }

                        cg::sync(cta);

                        if (cta.thread_rank() == 0) {
                            remote_am_done_writing_to_bottom_neighbor[next_iter_tile_flag_idx] =
                                iter + 1;
                        }
                    }
                }
            }
        }

        real *temp_pointer_first = a_new;
        a_new = a;
        a = temp_pointer_first;

        iter++;

        temp_iter_mod = cur_iter_mod;
        cur_iter_mod = next_iter_mod;
        next_iter_mod = temp_iter_mod;

        cg::sync(grid);

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            iteration_done[0] = iter;
        }

        cg::sync(grid);
    }
}
}  // namespace MultiGPUPeerTiling

int MultiGPUPeerTiling::init(int argc, char *argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);
    const bool compare_to_single_gpu = get_arg(argv, argv + argc, "-compare");

    real *a[MAX_NUM_DEVICES];
    real *a_new[MAX_NUM_DEVICES];
    int iy_end[MAX_NUM_DEVICES];

    real *halo_buffer_for_top_neighbor[MAX_NUM_DEVICES];
    real *halo_buffer_for_bottom_neighbor[MAX_NUM_DEVICES];

    int *is_top_done_computing_flags[MAX_NUM_DEVICES];
    int *is_bottom_done_computing_flags[MAX_NUM_DEVICES];

    real *a_ref_h;
    real *a_h;

    double runtime_serial_non_persistent = 0.0;

    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

#pragma omp parallel num_threads(num_devices)
    {
        int dev_id = omp_get_thread_num();

        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaFree(nullptr));

        if (compare_to_single_gpu && 0 == dev_id) {
            CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * sizeof(real)));
            CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * sizeof(real)));

            runtime_serial_non_persistent = single_gpu(nx, ny, iter_max, a_ref_h, 0, true);
        }

#pragma omp barrier

        int chunk_size;
        int chunk_size_low = (ny - 2) / num_devices;
        int chunk_size_high = chunk_size_low + 1;

        int height_per_gpu = ny / num_devices;

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

        int *iteration_done_flags[2];

        CUDA_RT_CALL(cudaMalloc(iteration_done_flags, 2 * sizeof(int)));
        CUDA_RT_CALL(cudaMalloc(iteration_done_flags, 2 * sizeof(int)));

        CUDA_RT_CALL(cudaMalloc(iteration_done_flags + 1, 2 * sizeof(int)));
        CUDA_RT_CALL(cudaMalloc(iteration_done_flags + 1, 2 * sizeof(int)));

        CUDA_RT_CALL(cudaMemset(iteration_done_flags[0], 0, 2 * sizeof(int)));
        CUDA_RT_CALL(cudaMemset(iteration_done_flags[1], 0, 2 * sizeof(int)));

        CUDA_RT_CALL(cudaMalloc(a + dev_id, nx * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(a_new + dev_id, nx * (chunk_size + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMemset(a[dev_id], 0, nx * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(a_new[dev_id], 0, nx * (chunk_size + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMalloc(halo_buffer_for_top_neighbor + dev_id, 2 * nx * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(halo_buffer_for_bottom_neighbor + dev_id, 2 * nx * sizeof(real)));

        CUDA_RT_CALL(cudaMemset(halo_buffer_for_top_neighbor[dev_id], 0, 2 * nx * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(halo_buffer_for_bottom_neighbor[dev_id], 0, 2 * nx * sizeof(real)));

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
        int numSms = deviceProp.multiProcessorCount - 2;

        dim3 dim_grid(numSms, 1, 1);
        dim3 dim_block(dim_block_x, dim_block_y);

        void *kernelArgsInner[] = {(void *)&a_new[dev_id],
                                   (void *)&a[dev_id],
                                   (void *)&iy_start,
                                   (void *)&iy_end[dev_id],
                                   (void *)&nx,
                                   (void *)&TILE_SIZE,
                                   (void *)&num_tiles_x,
                                   (void *)&num_tiles_y,
                                   (void *)&iy_end[top],
                                   (void *)&iy_start_bottom,
                                   (void *)&iter_max,
                                   (void *)&halo_buffer_for_top_neighbor[dev_id],
                                   (void *)&halo_buffer_for_bottom_neighbor[dev_id],
                                   (void *)&halo_buffer_for_bottom_neighbor[top],
                                   (void *)&halo_buffer_for_top_neighbor[bottom],
                                   (void *)&is_top_done_computing_flags[dev_id],
                                   (void *)&is_bottom_done_computing_flags[dev_id],
                                   (void *)&is_bottom_done_computing_flags[top],
                                   (void *)&is_top_done_computing_flags[bottom],
                                   (void *)&iteration_done_flags[0]};

        void *kernelArgsBoundary[] = {(void *)&a_new[dev_id],
                                      (void *)&a[dev_id],
                                      (void *)&iy_start,
                                      (void *)&iy_end[dev_id],
                                      (void *)&nx,
                                      (void *)&TILE_SIZE,
                                      (void *)&num_tiles_x,
                                      (void *)&num_tiles_y,
                                      (void *)&iy_end[top],
                                      (void *)&iy_start_bottom,
                                      (void *)&iter_max,
                                      (void *)&halo_buffer_for_top_neighbor[dev_id],
                                      (void *)&halo_buffer_for_bottom_neighbor[dev_id],
                                      (void *)&halo_buffer_for_bottom_neighbor[top],
                                      (void *)&halo_buffer_for_top_neighbor[bottom],
                                      (void *)&is_top_done_computing_flags[dev_id],
                                      (void *)&is_bottom_done_computing_flags[dev_id],
                                      (void *)&is_bottom_done_computing_flags[top],
                                      (void *)&is_top_done_computing_flags[bottom],
                                      (void *)&iteration_done_flags[0]};

#pragma omp barrier
        double start = omp_get_wtime();

        cudaStream_t inner_domain_stream;
        cudaStream_t boundary_sync_stream;

        CUDA_RT_CALL(cudaStreamCreate(&inner_domain_stream));
        CUDA_RT_CALL(cudaStreamCreate(&boundary_sync_stream));

        // THE KERNELS ARE SERIALIZED!
        // perhaps only on V100
        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *)MultiGPUPeerTiling::jacobi_kernel,
                                                 dim_grid, dim_block, kernelArgsInner, 0,
                                                 inner_domain_stream));

        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *)MultiGPUPeerTiling::boundary_sync_kernel,
                                                 2, dim_block, kernelArgsBoundary, 0,
                                                 boundary_sync_stream));

        CUDA_RT_CALL(cudaDeviceSynchronize());

        // Need to swap pointers on CPU if iteration count is odd
        // Technically, we don't know the iteration number (since we'll be doing l2-norm)
        // Could write iter to CPU when kernel is done
        if (iter_max % 2 == 1) {
            std::swap(a_new[dev_id], a[dev_id]);
        }

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