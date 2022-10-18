/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */
#include <cmath>
#include <cstdio>
#include <iostream>

#include <omp.h>

#include <cooperative_groups.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "../../include/common.h"
#include "../../include/multi-stream_nvshmem/multi-gpu-peer-tiling.cuh"

namespace cg = cooperative_groups;

namespace MultiGPUPeerTilingNvshmem
{
    __global__ void __launch_bounds__(1024, 1)
        jacobi_kernel(real *a_new, real *a, const int iz_start, const int iz_end, const int ny, const int nx,
                      const int comp_tile_size_x, const int comp_tile_size_y, const int comp_tile_size_z,
                      const int num_comp_tiles_x, const int num_comp_tiles_y, const int num_comp_tiles_z, const int iter_max,
                      volatile real *local_halo_buffer_for_top_neighbor,
                      volatile real *local_halo_buffer_for_bottom_neighbor,
                      volatile real *remote_my_halo_buffer_on_top_neighbor,
                      volatile real *remote_my_halo_buffer_on_bottom_neighbor,
                      volatile int *local_is_top_neighbor_done_writing_to_me,
                      volatile int *local_is_bottom_neighbor_done_writing_to_me,
                      volatile int *remote_am_done_writing_to_top_neighbor,
                      volatile int *remote_am_done_writing_to_bottom_neighbor,
                      volatile int *iteration_done)
    {
        cg::thread_block cta = cg::this_thread_block();
        cg::grid_group grid = cg::this_grid();

        const int grid_dim_x = (comp_tile_size_x + blockDim.x - 1) / blockDim.x;
        const int grid_dim_y = (comp_tile_size_y + blockDim.y - 1) / blockDim.y;

        const int block_idx_z = blockIdx.x / (grid_dim_x * grid_dim_y);
        const int block_idx_y = (blockIdx.x % (grid_dim_x * grid_dim_y)) / grid_dim_x;
        const int block_idx_x = blockIdx.x % grid_dim_x;

        int base_iz = block_idx_z * blockDim.z + threadIdx.z;
        int base_iy = block_idx_y * blockDim.y + threadIdx.y;
        int base_ix = block_idx_x * blockDim.x + threadIdx.x;

        int iter = 0;

        int cur_iter_mod = 0;
        int next_iter_mod = 1;
        int temp_iter_mod = 0;

        int iz;
        int iz_below;
        int iz_above;
        int iy;
        int iy_below;
        int iy_above;
        int ix;

        while (iter < iter_max)
        {
            for (iz = (base_iz + iz_start + 1) * ny * nx; iz < (iz_end - 1) * ny * nx; iz += comp_tile_size_z * ny * nx)
            {
                iz_below = iz + ny * nx;
                iz_above = iz - ny * nx;
                for (iy = (base_iy + 1) * nx; iy < (ny - 1) * nx; iy += comp_tile_size_y * nx)
                {
                    iy_below = iy + nx;
                    iy_above = iy - nx;
                    for (ix = (base_ix + 1); ix < (nx - 1); ix += comp_tile_size_x)
                    {
                        // big bottleneck here
                        const real new_val = (a[iz + iy + ix + 1] +
                                              a[iz + iy + ix - 1] +
                                              a[iz + iy_below + ix] +
                                              a[iz + iy_above + ix] +
                                              a[iz_below + iy + ix] +
                                              a[iz_above + iy + ix]) /
                                             real(6.0);

                        a_new[iz + iy + ix] = new_val;
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

            if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            {
                while (iteration_done[0] != iter)
                {
                }
                iteration_done[1] = iter;
            }

            cg::sync(grid);
        }
    }

    __global__ void __launch_bounds__(1024, 1)
        boundary_sync_kernel(real *a_new, real *a, const int iz_start, const int iz_end, const int ny, const int nx,
                             const int comm_tile_size_x, const int comm_tile_size_y,
                             const int num_comm_tiles_x, const int num_comm_tiles_y,
                             const int iter_max,
                             volatile real *local_halo_buffer_for_top_neighbor,
                             volatile real *local_halo_buffer_for_bottom_neighbor,
                             volatile real *remote_my_halo_buffer_on_top_neighbor,
                             volatile real *remote_my_halo_buffer_on_bottom_neighbor,
                             volatile int *local_is_top_neighbor_done_writing_to_me,
                             volatile int *local_is_bottom_neighbor_done_writing_to_me,
                             volatile int *remote_am_done_writing_to_top_neighbor,
                             volatile int *remote_am_done_writing_to_bottom_neighbor,
                             volatile int *iteration_done)
    {
        cg::thread_block cta = cg::this_thread_block();
        cg::grid_group grid = cg::this_grid();

        int num_flags = 2 * num_comm_tiles_x * num_comm_tiles_y;

        int iter = 0;

        int cur_iter_mod = 0;
        int next_iter_mod = 1;
        int temp_iter_mod = 0;

        int comm_tile_idx_y;
        int comm_tile_start_y;

        int comm_tile_idx_x;
        int comm_tile_start_x;

        int cur_iter_comm_tile_flag_idx_x;
        int cur_iter_comm_tile_flag_idx_y;
        int next_iter_comm_tile_flag_idx_x;
        int next_iter_comm_tile_flag_idx_y;

        int iy = 0;
        int ix = 0;

        while (iter < iter_max)
        {
            while (iteration_done[1] != iter)
            {
            }
            if (blockIdx.x == gridDim.x - 1)
            {
                for (comm_tile_idx_y = 0; comm_tile_idx_y < num_comm_tiles_y; comm_tile_idx_y++)
                {
                    comm_tile_start_y = (comm_tile_idx_y == 0) ? 1 : comm_tile_idx_y * comm_tile_size_y;

                    iy = threadIdx.z * blockDim.y + threadIdx.y + comm_tile_start_y;

                    for (comm_tile_idx_x = 0; comm_tile_idx_x < num_comm_tiles_x; comm_tile_idx_x++)
                    {
                        comm_tile_start_x = (comm_tile_idx_x == 0) ? 1 : comm_tile_idx_x * comm_tile_size_x;

                        ix = threadIdx.x + comm_tile_start_x;

                        if (cta.thread_rank() == 0)
                        {
                            cur_iter_comm_tile_flag_idx_x = comm_tile_idx_x;
                            cur_iter_comm_tile_flag_idx_y = comm_tile_idx_y;

                            while (local_is_top_neighbor_done_writing_to_me[cur_iter_comm_tile_flag_idx_y * num_comm_tiles_x +
                                                                            cur_iter_comm_tile_flag_idx_x +
                                                                            cur_iter_mod * num_flags] != iter)
                            {
                            }
                        }

                        cg::sync(cta);

                        if (iy < ny - 1 && ix < nx - 1)
                        {
                            const real first_row_val =
                                (a[iz_start * ny * nx + iy * nx + ix + 1] +
                                 a[iz_start * ny * nx + iy * nx + ix - 1] +
                                 a[iz_start * ny * nx + (iy + 1) * nx + ix] +
                                 a[iz_start * ny * nx + (iy - 1) * nx + ix] +
                                 a[(iz_start + 1) * ny * nx + iy * nx + ix] +
                                 remote_my_halo_buffer_on_top_neighbor[cur_iter_mod * ny * nx + iy * nx + ix]) /
                                real(6.0);

                            a_new[iz_start * ny * nx + iy * nx + ix] = first_row_val;
                            local_halo_buffer_for_top_neighbor[next_iter_mod * ny * nx + iy * nx + ix] = first_row_val;
                        }

                        cg::sync(cta);

                        if (cta.thread_rank() == 0)
                        {
                            next_iter_comm_tile_flag_idx_x = (num_comm_tiles_x + comm_tile_idx_x);
                            next_iter_comm_tile_flag_idx_y = (comm_tile_idx_y);

                            remote_am_done_writing_to_top_neighbor[next_iter_comm_tile_flag_idx_y * num_comm_tiles_x +
                                                                   next_iter_comm_tile_flag_idx_x +
                                                                   next_iter_mod * num_flags] = iter + 1;
                        }
                    }
                }
            }
            else if (blockIdx.x == gridDim.x - 2)
            {

                for (comm_tile_idx_y = 0; comm_tile_idx_y < num_comm_tiles_y; comm_tile_idx_y++)
                {
                    comm_tile_start_y = (comm_tile_idx_y == 0) ? 1 : comm_tile_idx_y * comm_tile_size_y;
                    iy = threadIdx.z * blockDim.y + threadIdx.y + comm_tile_start_y;

                    for (comm_tile_idx_x = 0; comm_tile_idx_x < num_comm_tiles_x; comm_tile_idx_x++)
                    {
                        comm_tile_start_x = (comm_tile_idx_x == 0) ? 1 : comm_tile_idx_x * comm_tile_size_x;
                        ix = threadIdx.x + comm_tile_start_x;

                        if (cta.thread_rank() == 0)
                        {
                            cur_iter_comm_tile_flag_idx_x = (num_comm_tiles_x + comm_tile_idx_x);
                            cur_iter_comm_tile_flag_idx_y = (comm_tile_idx_y);
                            while (
                                local_is_bottom_neighbor_done_writing_to_me[cur_iter_comm_tile_flag_idx_y * num_comm_tiles_x +
                                                                            cur_iter_comm_tile_flag_idx_x +
                                                                            cur_iter_mod * num_flags] != iter)
                            {
                            }
                        }

                        cg::sync(cta);

                        if (iy < ny - 1 && ix < nx - 1)
                        {

                            const real last_row_val =
                                (a[(iz_end - 1) * ny * nx + iy * nx + ix + 1] +
                                 a[(iz_end - 1) * ny * nx + iy * nx + ix - 1] +
                                 a[(iz_end - 1) * ny * nx + (iy + 1) * nx + ix] +
                                 a[(iz_end - 1) * ny * nx + (iy - 1) * nx + ix] +
                                 remote_my_halo_buffer_on_bottom_neighbor[cur_iter_mod * ny * nx + iy * nx + ix] +
                                 a[(iz_end - 2) * ny * nx + iy * nx + ix]) /
                                real(6.0);

                            a_new[(iz_end - 1) * ny * nx + iy * nx + ix] = last_row_val;
                            local_halo_buffer_for_bottom_neighbor[next_iter_mod * ny * nx + iy * nx + ix] = last_row_val;
                        }

                        cg::sync(cta);

                        if (cta.thread_rank() == 0)
                        {
                            next_iter_comm_tile_flag_idx_x = comm_tile_idx_x;
                            next_iter_comm_tile_flag_idx_y = comm_tile_idx_y;

                            remote_am_done_writing_to_bottom_neighbor[next_iter_comm_tile_flag_idx_y * num_comm_tiles_x +
                                                                      next_iter_comm_tile_flag_idx_x +
                                                                      next_iter_mod * num_flags] = iter + 1;
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

            if (threadIdx.x == 0 && threadIdx.y == 0)
            {
                iteration_done[0] = iter;
            }

            cg::sync(grid);
        }
    }
} // namespace MultiGPUPeerTiling

int MultiGPUPeerTilingNvshmem::init(int argc, char *argv[])
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
        CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, dev_id));
        int numSms = deviceProp.multiProcessorCount;

        constexpr int dim_block_x = 32;
        constexpr int dim_block_y = 32;
        constexpr int dim_block_z = 1;

        constexpr int comp_tile_size_x = dim_block_x;
        constexpr int comp_tile_size_y = dim_block_y;
        int comp_tile_size_z;

        constexpr int comm_tile_size_x = dim_block_x;
        constexpr int comm_tile_size_y = dim_block_z * dim_block_y;

        constexpr int grid_dim_x = (comp_tile_size_x + dim_block_x - 1) / dim_block_x;
        constexpr int grid_dim_y = (comp_tile_size_y + dim_block_y - 1) / dim_block_y;

        int max_thread_blocks_z = (numSms - 2) / (grid_dim_x * grid_dim_y);

        comp_tile_size_z = dim_block_z * max_thread_blocks_z;

        int num_comp_tiles_x = nx / comp_tile_size_x + (nx % comp_tile_size_x != 0);
        int num_comp_tiles_y = ny / comp_tile_size_y + (ny % comp_tile_size_y != 0);
        int num_comp_tiles_z = nz_per_gpu / comp_tile_size_z + (nz_per_gpu % comp_tile_size_z != 0);

        int num_comm_tiles_x = nx / comm_tile_size_x + (nx % comm_tile_size_x != 0);
        int num_comm_tiles_y = ny / comm_tile_size_y + (ny % comm_tile_size_y != 0);

        int total_num_flags = 4 * num_comm_tiles_x * num_comm_tiles_y;

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

        int *iteration_done_flags[2];

        CUDA_RT_CALL(cudaMalloc(iteration_done_flags, 2 * sizeof(int)));
        CUDA_RT_CALL(cudaMalloc(iteration_done_flags, 2 * sizeof(int)));

        CUDA_RT_CALL(cudaMalloc(iteration_done_flags + 1, 2 * sizeof(int)));
        CUDA_RT_CALL(cudaMalloc(iteration_done_flags + 1, 2 * sizeof(int)));

        CUDA_RT_CALL(cudaMemset(iteration_done_flags[0], 0, 2 * sizeof(int)));
        CUDA_RT_CALL(cudaMemset(iteration_done_flags[1], 0, 2 * sizeof(int)));

        CUDA_RT_CALL(cudaMalloc(a + dev_id, nx * ny * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(a_new + dev_id, nx * ny * (chunk_size + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMemset(a[dev_id], 0, nx * ny * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(a_new[dev_id], 0, nx * ny * (chunk_size + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMalloc(halo_buffer_for_top_neighbor + dev_id, 2 * nx * ny * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(halo_buffer_for_bottom_neighbor + dev_id, 2 * nx * ny * sizeof(real)));

        CUDA_RT_CALL(cudaMemset(halo_buffer_for_top_neighbor[dev_id], 0, 2 * nx * ny * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(halo_buffer_for_bottom_neighbor[dev_id], 0, 2 * nx * ny * sizeof(real)));

        CUDA_RT_CALL(cudaMalloc(is_top_done_computing_flags + dev_id, total_num_flags * sizeof(int)));
        CUDA_RT_CALL(cudaMalloc(is_bottom_done_computing_flags + dev_id, total_num_flags * sizeof(int)));

        CUDA_RT_CALL(cudaMemset(is_top_done_computing_flags[dev_id], 0, total_num_flags * sizeof(int)));
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
            iz_start_global = num_ranks_low * chunk_size_low + (dev_id - num_ranks_low) * chunk_size_high + 1;
        }
        int iz_end_global = iz_start_global + chunk_size - 1; // My last index in the global array

        int iz_start = 1;
        iz_end[dev_id] = (iz_end_global - iz_start_global + 1) + iz_start;

        // Set diriclet boundary conditions on left and right border
        initialize_boundaries<<<(nz / num_devices) / 128 + 1, 128>>>(
            a[dev_id], a_new[dev_id], PI, iz_start_global - 1, nx, ny, chunk_size + 2, nz);
        CUDA_RT_CALL(cudaGetLastError());

        CUDA_RT_CALL(cudaDeviceSynchronize());

        dim3 dim_grid(numSms - 2, 1, 1);
        dim3 dim_block(dim_block_x, dim_block_y, dim_block_z);

        void *kernelArgsInner[] = {(void *)&a_new[dev_id],
                                   (void *)&a[dev_id],
                                   (void *)&iz_start,
                                   (void *)&iz_end[dev_id],
                                   (void *)&ny,
                                   (void *)&nx,
                                   (void *)&comp_tile_size_x,
                                   (void *)&comp_tile_size_y,
                                   (void *)&comp_tile_size_z,
                                   (void *)&num_comp_tiles_x,
                                   (void *)&num_comp_tiles_y,
                                   (void *)&num_comp_tiles_z,
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
                                      (void *)&iz_start,
                                      (void *)&iz_end[dev_id],
                                      (void *)&ny,
                                      (void *)&nx,
                                      (void *)&comm_tile_size_x,
                                      (void *)&comm_tile_size_y,
                                      (void *)&num_comm_tiles_x,
                                      (void *)&num_comm_tiles_y,
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
        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *)MultiGPUPeerTilingNvshmem::jacobi_kernel,
                                                 dim_grid, dim_block, kernelArgsInner, 0,
                                                 inner_domain_stream));

        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *)MultiGPUPeerTilingNvshmem::boundary_sync_kernel,
                                                 2, dim_block, kernelArgsBoundary, 0,
                                                 boundary_sync_stream));

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
            CUDA_RT_CALL(cudaMemcpy(
                a_h + iz_start_global * ny * nx, a[dev_id] + ny * nx,
                std::min((nz - iz_start_global) * ny * nx, chunk_size * nx * ny) * sizeof(real),
                cudaMemcpyDeviceToHost));
        }

#pragma omp barrier

#pragma omp master
        {
            report_results(nz, ny, nx, a_ref_h, a_h, num_devices, runtime_serial_non_persistent, start, stop,
                           compare_to_single_gpu);
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
