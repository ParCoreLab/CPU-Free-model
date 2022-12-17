/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */
#include <cmath>
#include <cstdio>
#include <iostream>

#include <omp.h>

#include <cooperative_groups.h>

#include "../../include/common.h"
#include "../../include/single-stream/multi-threaded-one-block-comm.cuh"

namespace cg = cooperative_groups;

namespace SSMultiThreadedOneBlockComm
{
    __global__ void __launch_bounds__(1024, 1)
        jacobi_kernel(real *a_new, real *a, const int iy_start, const int iy_end, const int nx,
                       const int num_comm_tiles,
                      const int iter_max, volatile real *local_halo_buffer_for_top_neighbor,
                      volatile real *local_halo_buffer_for_bottom_neighbor,
                      volatile real *remote_my_halo_buffer_on_top_neighbor,
                      volatile real *remote_my_halo_buffer_on_bottom_neighbor,
                      volatile int *local_is_top_neighbor_done_writing_to_me,
                      volatile int *local_is_bottom_neighbor_done_writing_to_me,
                      volatile int *remote_am_done_writing_to_top_neighbor,
                      volatile int *remote_am_done_writing_to_bottom_neighbor)
    {
        cg::thread_block cta = cg::this_thread_block();
        cg::grid_group grid = cg::this_grid();

        int iter = 0;

        int cur_iter_mod = 0;
        int next_iter_mod = 1;

        const int num_flags = 2 * num_comm_tiles;

        while (iter < iter_max)
        {
            if (blockIdx.x == gridDim.x - 1)
            {
                int ix = threadIdx.y * blockDim.x + threadIdx.x + 1;
                for (int comm_tile_idx = 0; comm_tile_idx < num_comm_tiles; comm_tile_idx++, ix += blockDim.x * blockDim.y)
                {
                    if (!cta.thread_rank())
                    {
                        int cur_iter_comm_tile_flag_idx = comm_tile_idx + cur_iter_mod * num_flags;

                        while (local_is_top_neighbor_done_writing_to_me[cur_iter_comm_tile_flag_idx] !=
                               iter)
                        {
                        }
                    }

                    cg::sync(cta);

                    if (ix < (nx - 1))
                    {
                        const real first_row_val =
                            0.25 * (a[iy_start * nx + ix + 1] + a[iy_start * nx + ix - 1] +
                                    a[iy_start * nx + nx + ix] +
                                    remote_my_halo_buffer_on_top_neighbor[nx * cur_iter_mod + ix]);

                        a_new[iy_start * nx + ix] = first_row_val;
                        local_halo_buffer_for_top_neighbor[nx * next_iter_mod + ix] = first_row_val;
                    }

                    cg::sync(cta);

                    if (!cta.thread_rank())
                    {
                        int next_iter_comm_tile_flag_idx =
                            (num_comm_tiles + comm_tile_idx) + next_iter_mod * num_flags;

                        remote_am_done_writing_to_top_neighbor[next_iter_comm_tile_flag_idx] = iter + 1;
                    }

                    cg::sync(cta);

                    if (!cta.thread_rank())
                    {
                        int cur_iter_comm_tile_flag_idx =
                            (num_comm_tiles + comm_tile_idx) + cur_iter_mod * num_flags;

                        while (
                            local_is_bottom_neighbor_done_writing_to_me[cur_iter_comm_tile_flag_idx] !=
                            iter)
                        {
                        }
                    }

                    cg::sync(cta);

                    if (ix < (nx - 1))
                    {
                        const real last_row_val =
                            0.25 * (a[(iy_end - 1) * nx + ix + 1] + a[(iy_end - 1) * nx + ix - 1] +
                                    remote_my_halo_buffer_on_bottom_neighbor[nx * cur_iter_mod + ix] +
                                    a[(iy_end - 1) * nx - nx + ix]);

                        a_new[(iy_end - 1) * nx + ix] = last_row_val;
                        local_halo_buffer_for_bottom_neighbor[nx * next_iter_mod + ix] = last_row_val;
                    }

                    cg::sync(cta);

                    if (!cta.thread_rank())
                    {
                        int next_iter_comm_tile_flag_idx = comm_tile_idx + next_iter_mod * num_flags;

                        remote_am_done_writing_to_bottom_neighbor[next_iter_comm_tile_flag_idx] =
                            iter + 1;
                    }
                }
            }
            else
            {
                for (int iy = (blockIdx.x * blockDim.y + threadIdx.y + iy_start + 1) * nx;
                 iy < (iy_end - 1) * nx; iy += (gridDim.x - 1) * blockDim.y * nx)
                {
                    for (int ix = threadIdx.x + 1; ix < (nx - 1); ix += blockDim.x)
                    {
                        const real new_val = 0.25 * (a[iy + ix + 1] + a[iy + ix - 1] +
                                                     a[iy + nx + ix] + a[iy - nx + ix]);
                        a_new[iy + ix] = new_val;
                    }
                }
            }

            real *temp_pointer = a_new;
            a_new = a;
            a = temp_pointer;

            iter++;

            next_iter_mod = cur_iter_mod;
            cur_iter_mod = 1 - cur_iter_mod;

            cg::sync(grid);
        }
    }
} // namespace SSMultiThreadedOneBlockComm

int SSMultiThreadedOneBlockComm::init(int argc, char *argv[])
{
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

        if (compare_to_single_gpu && 0 == dev_id)
        {
            CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * sizeof(real)));
            CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * sizeof(real)));

            runtime_serial_non_persistent = single_gpu(nx, ny, iter_max, a_ref_h, 0, true);
        }

#pragma omp barrier

        int chunk_size;
        int chunk_size_low = (ny - 2) / num_devices;
        int chunk_size_high = chunk_size_low + 1;

        int height_per_gpu = ny / num_devices;

        cudaDeviceProp deviceProp{};
        CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, dev_id));
        int numSms = deviceProp.multiProcessorCount;

        constexpr int dim_block_x = 32;
        constexpr int dim_block_y = 32;

        // int comp_tile_size_x = 256;

        //int grid_dim_x = 8;
        //int grid_dim_y = (numSms - 1) / grid_dim_x;

        // int comp_tile_size_y = dim_block_y * grid_dim_y;

        // printf("Computation tile dimensions: %dx%d\n", comp_tile_size_x, comp_tile_size_y);

        // int num_comp_tiles_x = nx / comp_tile_size_x + (nx % comp_tile_size_x != 0);
        // int num_comp_tiles_y =
        //     height_per_gpu / comp_tile_size_y + (height_per_gpu % comp_tile_size_y != 0);

        int comm_tile_size = dim_block_x * dim_block_y;
        int num_comm_tiles = nx / comm_tile_size + (nx % comm_tile_size != 0);
        int num_flags = 4 * num_comm_tiles;

        // printf("Number of communication tiles: %d\n", num_comm_tiles);

        int num_ranks_low = num_devices * chunk_size_low + num_devices - (ny - 2);
        if (dev_id < num_ranks_low)
            chunk_size = chunk_size_low;
        else
            chunk_size = chunk_size_high;

        const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);
        const int bottom = (dev_id + 1) % num_devices;

        if (top != dev_id)
        {
            int canAccessPeer = 0;
            CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, top));
            if (canAccessPeer)
            {
                CUDA_RT_CALL(cudaDeviceEnablePeerAccess(top, 0));
            }
            else
            {
                std::cerr << "P2P access required from " << dev_id << " to " << top << std::endl;
            }
            if (top != bottom)
            {
                canAccessPeer = 0;
                CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, bottom));
                if (canAccessPeer)
                {
                    CUDA_RT_CALL(cudaDeviceEnablePeerAccess(bottom, 0));
                }
                else
                {
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
        int iy_start_global; // My start index in the global array
        if (dev_id < num_ranks_low)
        {
            iy_start_global = dev_id * chunk_size_low + 1;
        }
        else
        {
            iy_start_global =
                num_ranks_low * chunk_size_low + (dev_id - num_ranks_low) * chunk_size_high + 1;
        }
        int iy_end_global = iy_start_global + chunk_size - 1; // My last index in the global array

        int iy_start = 1;
        iy_end[dev_id] = (iy_end_global - iy_start_global + 1) + iy_start;

        // Set diriclet boundary conditions on left and right border
        initialize_boundaries<<<(ny / num_devices) / 128 + 1, 128>>>(
            a[dev_id], a_new[dev_id], PI, iy_start_global - 1, nx, chunk_size + 2, ny);
        CUDA_RT_CALL(cudaGetLastError());

        CUDA_RT_CALL(cudaDeviceSynchronize());

        dim3 dim_grid(numSms);
        dim3 dim_block(dim_block_x, dim_block_y);

        void *kernelArgs[] = {(void *)&a_new[dev_id],
                              (void *)&a[dev_id],
                              (void *)&iy_start,
                              (void *)&iy_end[dev_id],
                              (void *)&nx,
                              (void *)&num_comm_tiles,
                              (void *)&iter_max,
                              (void *)&halo_buffer_for_top_neighbor[dev_id],
                              (void *)&halo_buffer_for_bottom_neighbor[dev_id],
                              (void *)&halo_buffer_for_bottom_neighbor[top],
                              (void *)&halo_buffer_for_top_neighbor[bottom],
                              (void *)&is_top_done_computing_flags[dev_id],
                              (void *)&is_bottom_done_computing_flags[dev_id],
                              (void *)&is_bottom_done_computing_flags[top],
                              (void *)&is_top_done_computing_flags[bottom]};

#pragma omp barrier
        double start = omp_get_wtime();

        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *)SSMultiThreadedOneBlockComm::jacobi_kernel,
                                                 dim_grid, dim_block, kernelArgs, 0, nullptr));

        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());

        // Need to swap pointers on CPU if iteration count is odd
        // Technically, we don't know the iteration number (since we'll be doing l2-norm)
        // Could write iter to CPU when kernel is done
        if (iter_max % 2 == 1)
        {
            std::swap(a_new[dev_id], a[dev_id]);
        }

#pragma omp barrier
        double stop = omp_get_wtime();

        if (compare_to_single_gpu)
        {
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
        CUDA_RT_CALL(cudaFree(halo_buffer_for_top_neighbor[dev_id]));
        CUDA_RT_CALL(cudaFree(halo_buffer_for_bottom_neighbor[dev_id]));
        CUDA_RT_CALL(cudaFree(is_top_done_computing_flags[dev_id]));
        CUDA_RT_CALL(cudaFree(is_bottom_done_computing_flags[dev_id]));

        if (compare_to_single_gpu && 0 == dev_id)
        {
            CUDA_RT_CALL(cudaFreeHost(a_h));
            CUDA_RT_CALL(cudaFreeHost(a_ref_h));
        }
    }

    return 0;
}