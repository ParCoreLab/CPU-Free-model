/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */
#include <cmath>
#include <cstdio>
#include <iostream>

#include <omp.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include "../../include/single-stream/multi-threaded-one-block-comm-layer.cuh"

namespace cg = cooperative_groups;
// Plan 1 turn comm remote acesseses to local, after a iteration finishes, use every thread in grid
// to copy halos to top and bottom sections.
namespace SSMultiThreadedOneBlockCommLayer
{

    __global__ void __launch_bounds__(1024, 1)
        jacobi_kernel(real *a_new, real *a, const int iz_start, const int iz_end, const int ny,
                      const int nx, const int grid_dim_y, const int grid_dim_x, const int iter_max,
                      volatile real *local_halo_buffer_for_top_neighbor,
                      volatile real *local_halo_buffer_for_bottom_neighbor,
                      volatile const real *remote_my_halo_buffer_on_top_neighbor,
                      volatile const real *remote_my_halo_buffer_on_bottom_neighbor,
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

        const int comp_size_iz = ((gridDim.x - 1) / (grid_dim_y * grid_dim_x)) * blockDim.z * ny * nx;
        const int comp_size_iy = grid_dim_y * blockDim.y * nx;
        const int comp_size_ix = grid_dim_x * blockDim.x;

        const int comp_start_iz = ((blockIdx.x / (grid_dim_y * grid_dim_x)) * blockDim.z + threadIdx.z + iz_start + 1) * ny * nx;
        const int comp_start_iy = ((blockIdx.x / grid_dim_x % grid_dim_y) * blockDim.y + threadIdx.y + 1) * nx;
        const int comp_start_ix = ((blockIdx.x % grid_dim_x) * blockDim.x + threadIdx.x + 1);

        const int end_iz = (iz_end - 1) * ny * nx;
        const int end_iy = (ny - 1) * nx;
        const int end_ix = (nx - 1);

        const int comm_size_iy = blockDim.y * blockDim.z * nx;
        const int comm_size_ix = blockDim.x;

        const int comm_start_iy = (threadIdx.z * blockDim.y + threadIdx.y + 1) * nx;
        const int comm_start_ix = threadIdx.x + 1;
        const int comm_start_iz = iz_start * ny * nx ;

        while (iter < iter_max)
        {
            if (blockIdx.x == gridDim.x - 1)
            {
                if (cta.thread_rank() == 0)
                {
                    while (local_is_top_neighbor_done_writing_to_me[cur_iter_mod * 2] !=
                           iter)
                    {
                    }
                }
                cg::sync(cta);
                for (int iy = comm_start_iy; iy < end_iy; iy += comm_size_iy)
                {
                    for (int ix = comm_size_ix; ix < end_ix; ix += comm_size_ix)
                    {
                        const real first_row_val = (real(1) / real(6)) * (a[comm_start_iz + iy + ix + 1] +
                                                                          a[comm_start_iz + iy + ix - 1] +
                                                                          a[comm_start_iz + iy + nx + ix] +
                                                                          a[comm_start_iz + iy - nx + ix] +
                                                                          a[comm_start_iz + ny * nx + iy + ix] +
                                                                          remote_my_halo_buffer_on_top_neighbor[cur_iter_mod * ny * nx +
                                                                                                                iy + ix]);
                        a_new[comm_start_iz + iy + ix] = first_row_val;
                        local_halo_buffer_for_top_neighbor[next_iter_mod * ny * nx + iy + ix] =
                            first_row_val;
                    }
                }
                cg::sync(cta);
                if (cta.thread_rank() == 0)
                {
                    remote_am_done_writing_to_top_neighbor[next_iter_mod * 2 + 1] = iter + 1;

                    while (
                        local_is_bottom_neighbor_done_writing_to_me[cur_iter_mod * 2 + 1] !=
                        iter)
                    {
                    }
                }
                cg::sync(cta);
                for (int iy = comm_start_iy; iy < end_iy; iy += comm_size_iy)
                {
                    for (int ix = comm_size_ix; ix < end_ix; ix += comm_size_ix)
                    {
                        const real last_row_val = (real(1) / real(6)) * (a[end_iz + iy + ix + 1] +
                                                                         a[end_iz + iy + ix - 1] +
                                                                         a[end_iz + iy + nx + ix] +
                                                                         a[end_iz + iy - nx + ix] +
                                                                         remote_my_halo_buffer_on_bottom_neighbor[cur_iter_mod * ny * nx +
                                                                                                                  iy + ix] +
                                                                         a[end_iz - ny * nx + iy + ix]);
                        a_new[end_iz + iy + ix] = last_row_val;
                        local_halo_buffer_for_bottom_neighbor[next_iter_mod * ny * nx + iy +
                                                              ix] = last_row_val;
                    }
                }
                cg::sync(cta);

                if (cta.thread_rank() == 0)
                {
                    remote_am_done_writing_to_bottom_neighbor[next_iter_mod * 2] =
                        iter + 1;
                }
            }
            else
            {
                for (int iz = comp_start_iz; iz < end_iz; iz += comp_size_iz)
                {
                    for (int iy = comp_start_iy; iy < end_iy; iy += comp_size_iy)
                    {
                        for (int ix = comp_start_ix; ix < end_ix; ix += comp_size_ix)
                        {
                            a_new[iz + iy + ix] = (real(1) / real(6)) *
                                                  (a[iz + iy + ix + 1] + a[iz + iy + ix - 1] + a[iz + iy + nx + ix] +
                                                   a[iz + iy - nx + ix] + a[iz + ny * nx + iy + ix] +
                                                   a[iz - ny * nx + iy + ix]);
                        }
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
} // namespace SSMultiThreadedOneBlockCommLayer

int SSMultiThreadedOneBlockCommLayer::init(int argc, char *argv[])
{
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 512);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 512);
    const int nz = get_argval<int>(argv, argv + argc, "-nz", 512);
    const bool compare_to_single_gpu = get_arg(argv, argv + argc, "-compare");

    real *a[MAX_NUM_DEVICES];
    real *a_new[MAX_NUM_DEVICES];
    int iz_end[MAX_NUM_DEVICES];

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
            CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * nz * sizeof(real)));
            CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * nz * sizeof(real)));

            runtime_serial_non_persistent = single_gpu(nz, ny, nx, iter_max, a_ref_h, 0, true);
        }

#pragma omp barrier

        int chunk_size;
        int chunk_size_low = (nz - 2) / num_devices;
        int chunk_size_high = chunk_size_low + 1;

        int nz_per_gpu = nz / num_devices;

        cudaDeviceProp deviceProp{};
        int maxActiveBlocksPerSM = 0;
        CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, dev_id));
        CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocksPerSM, (void *)SSMultiThreadedOneBlockCommLayer::jacobi_kernel, 1024, 0));

        int numSms = deviceProp.multiProcessorCount * maxActiveBlocksPerSM;

        constexpr int dim_block_x = 32;
        constexpr int dim_block_y = 8;
        constexpr int dim_block_z = 4;

        // constexpr int comp_tile_size_x = dim_block_x;
        // constexpr int comp_tile_size_y = 8 * dim_block_y;
        // int comp_tile_size_z;

        // constexpr int comm_tile_size_x = dim_block_x;
        // constexpr int comm_tile_size_y = dim_block_z * dim_block_y;

        constexpr int grid_dim_x = 2;
        constexpr int grid_dim_y = 4;
        const int grid_dim_z = (numSms - 1) / (grid_dim_x * grid_dim_y);
        printf("Grid Dim: %dx%dx%d\n", grid_dim_x, grid_dim_y, grid_dim_z);
        // comp_tile_size_z = dim_block_z * grid_dim_z;

        // int num_comp_tiles_x = nx / comp_tile_size_x + (nx % comp_tile_size_x != 0);
        // int num_comp_tiles_y = ny / comp_tile_size_y + (ny % comp_tile_size_y != 0);
        // int num_comp_tiles_z = nz_per_gpu / comp_tile_size_z + (nz_per_gpu % comp_tile_size_z != 0);

        // int num_comm_tiles_x = nx / comm_tile_size_x + (nx % comm_tile_size_x != 0);
        // int num_comm_tiles_y = ny / comm_tile_size_y + (ny % comm_tile_size_y != 0);

        int total_num_flags = 4;

        int num_ranks_low = num_devices * chunk_size_low + num_devices - (nz - 2);
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

        CUDA_RT_CALL(cudaMalloc(a + dev_id, nx * ny * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(a_new + dev_id, nx * ny * (chunk_size + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMemset(a[dev_id], 0, nx * ny * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(a_new[dev_id], 0, nx * ny * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(halo_buffer_for_top_neighbor + dev_id, 2 * nx * ny * sizeof(real)));
        CUDA_RT_CALL(
            cudaMalloc(halo_buffer_for_bottom_neighbor + dev_id, 2 * nx * ny * sizeof(real)));

        CUDA_RT_CALL(
            cudaMemset(halo_buffer_for_top_neighbor[dev_id], 0, 2 * nx * ny * sizeof(real)));
        CUDA_RT_CALL(
            cudaMemset(halo_buffer_for_bottom_neighbor[dev_id], 0, 2 * nx * ny * sizeof(real)));

        CUDA_RT_CALL(
            cudaMalloc(is_top_done_computing_flags + dev_id, total_num_flags * sizeof(int)));
        CUDA_RT_CALL(
            cudaMalloc(is_bottom_done_computing_flags + dev_id, total_num_flags * sizeof(int)));

        CUDA_RT_CALL(
            cudaMemset(is_top_done_computing_flags[dev_id], 0, total_num_flags * sizeof(int)));
        CUDA_RT_CALL(
            cudaMemset(is_bottom_done_computing_flags[dev_id], 0, total_num_flags * sizeof(int)));

        // Calculate local domain boundaries
        int iz_start_global; // My start index in the global array
        if (dev_id < num_ranks_low)
        {
            iz_start_global = dev_id * chunk_size_low + 1;
        }
        else
        {
            iz_start_global =
                num_ranks_low * chunk_size_low + (dev_id - num_ranks_low) * chunk_size_high + 1;
        }
        int iz_end_global = iz_start_global + chunk_size - 1; // My last index in the global array

        int iz_start = 1;
        iz_end[dev_id] = (iz_end_global - iz_start_global + 1) + iz_start;

        initialize_boundaries<<<(nz / num_devices) / 128 + 1, 128>>>(
            a_new[dev_id], a[dev_id], PI, iz_start_global - 1, nx, ny, chunk_size + 2, nz);
        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());

        dim3 dim_grid(grid_dim_x * grid_dim_y * grid_dim_z + 1);
        dim3 dim_block(dim_block_x, dim_block_y, dim_block_z);

        void *kernelArgs[] = {(void *)&a_new[dev_id],
                              (void *)&a[dev_id],
                              (void *)&iz_start,
                              (void *)&iz_end[dev_id],
                              (void *)&ny,
                              (void *)&nx,
                              (void *)&grid_dim_y,
                              (void *)&grid_dim_x,
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

        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *)SSMultiThreadedOneBlockCommLayer::jacobi_kernel,
                                                 dim_grid, dim_block, kernelArgs, 0, nullptr));

        CUDA_RT_CALL(cudaDeviceSynchronize());
        CUDA_RT_CALL(cudaGetLastError());

        // Need to swap pointers on CPU if iteration count is odd
        // Technically, we don't know the iteration number (since we'll be doing
        // l2-norm) Could write iter to CPU when kernel is done
        if (iter_max % 2 == 1)
        {
            std::swap(a_new[dev_id], a[dev_id]);
        }

#pragma omp barrier
        double stop = omp_get_wtime();

        if (compare_to_single_gpu)
        {
            CUDA_RT_CALL(cudaMemcpy(
                a_h + iz_start_global * ny * nx, a[dev_id] + ny * nx,
                std::min((nz - iz_start_global) * ny * nx, chunk_size * nx * ny) * sizeof(real),
                cudaMemcpyDeviceToHost));
        }

#pragma omp barrier

#pragma omp master
        {
            report_results(nz, ny, nx, a_ref_h, a_h, num_devices, runtime_serial_non_persistent,
                           start, stop, compare_to_single_gpu);
        }

        CUDA_RT_CALL(cudaFree(a_new[dev_id]));
        CUDA_RT_CALL(cudaFree(a[dev_id]));

        if (compare_to_single_gpu && 0 == dev_id)
        {
            CUDA_RT_CALL(cudaFreeHost(a_h));
            CUDA_RT_CALL(cudaFreeHost(a_ref_h));
        }
    }

    return 0;
}
