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

namespace SSMultiThreadedOneBlockComm {
__global__ void __launch_bounds__(1024, 1)
    jacobi_kernel(real *a_new, real *a, const int iz_start, const int iz_end, const int ny,
                  const int nx, const int comp_tile_size_x, const int comp_tile_size_y,
                  const int comp_tile_size_z, const int comm_tile_size_x,
                  const int comm_tile_size_y, const int num_comp_tiles_x,
                  const int num_comp_tiles_y, const int num_comp_tiles_z,
                  const int num_comm_tiles_x, const int num_comm_tiles_y, const int iter_max,
                  volatile real *local_halo_buffer_for_top_neighbor,
                  volatile real *local_halo_buffer_for_bottom_neighbor,
                  volatile real *remote_my_halo_buffer_on_top_neighbor,
                  volatile real *remote_my_halo_buffer_on_bottom_neighbor,
                  volatile int *local_is_top_neighbor_done_writing_to_me,
                  volatile int *local_is_bottom_neighbor_done_writing_to_me,
                  volatile int *remote_am_done_writing_to_top_neighbor,
                  volatile int *remote_am_done_writing_to_bottom_neighbor) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    int grid_dim_x = (comp_tile_size_x + blockDim.x - 1) / blockDim.x;
    int grid_dim_y = (comp_tile_size_y + blockDim.y - 1) / blockDim.y;
    int block_idx_z = blockIdx.x / (grid_dim_x * grid_dim_y);
    int block_idx_y = blockIdx.x / grid_dim_x;
    int block_idx_x = blockIdx.x % grid_dim_x;

    int base_iz = block_idx_z * blockDim.z + threadIdx.z;
    int base_iy = block_idx_y * blockDim.y + threadIdx.y;
    int base_ix = block_idx_x * blockDim.x + threadIdx.x;

    int iter = 0;

    int cur_iter_mod = 0;
    int next_iter_mod = 1;

    while (iter < iter_max) {
        if (blockIdx.x == gridDim.x - 1) {
            int num_flags = 2 * num_comm_tiles_x * num_comm_tiles_y;
            for (int comm_tile_idy = 0; comm_tile_idy < num_comm_tiles_y; comm_tile_idy++) {
                int comm_tile_start_y = (comm_tile_idy == 0) ? 1 : comm_tile_idy * comm_tile_size_y;
                int comm_tile_end_y = (comm_tile_idy == (num_comm_tiles_y - 1))
                                          ? ny - 1
                                          : (comm_tile_idy + 1) * comm_tile_size_y;
                int iy = threadIdx.z * blockDim.y + threadIdx.y + comm_tile_start_y;
                for (int comm_tile_idx = 0; comm_tile_idx < num_comm_tiles_x; comm_tile_idx++) {
                    int comm_tile_start_x =
                        (comm_tile_idx == 0) ? 1 : comm_tile_idx * comm_tile_size_x;
                    int comm_tile_end_x = (comm_tile_idx == (num_comm_tiles_x - 1))
                                              ? nx - 1
                                              : (comm_tile_idx + 1) * comm_tile_size_x;
                    int ix = threadIdx.y * blockDim.x + threadIdx.x + comm_tile_start_x;

                    if (cta.thread_rank() == 0) {
                        int cur_iter_comm_tile_flag_idx = comm_tile_idx + cur_iter_mod * num_flags;
                        int cur_iter_comm_tile_flag_idy = comm_tile_idy + cur_iter_mod * num_flags;

                        while (
                            local_is_top_neighbor_done_writing_to_me[cur_iter_comm_tile_flag_idy *
                                                                         nx +
                                                                     cur_iter_comm_tile_flag_idx] !=
                            iter) {
                        }
                    }

                    cg::sync(cta);

                    if (iy < comm_tile_end_y && ix < comm_tile_end_x) {
                        const real first_row_val =
                            (1 / 6) *
                            (a[iz_start * ny * nx + iy * nx + ix + 1] +
                             a[iz_start * ny * nx + iy * nx + ix - 1] +
                             a[iz_start * ny * nx + (iy + 1) * nx + ix] +
                             a[iz_start * ny * nx + (iy - 1) * nx + ix] +
                             a[(iz_start + 1) * ny * nx + iy * nx + ix] +
                             remote_my_halo_buffer_on_top_neighbor[cur_iter_mod * ny * nx +
                                                                   iy * nx + ix]);

                        a_new[iz_start * ny * nx + iy * nx + ix] = first_row_val;
                        local_halo_buffer_for_top_neighbor[next_iter_mod * ny * nx + iy * nx + ix] =
                            first_row_val;
                    }

                    cg::sync(cta);

                    if (cta.thread_rank() == 0) {
                        int next_iter_comm_tile_flag_idx =
                            (num_comm_tiles_x + comm_tile_idx) + next_iter_mod * num_flags;

                        int next_iter_comm_tile_flag_idy =
                            (num_comm_tiles_y + comm_tile_idy) + next_iter_mod * num_flags;

                        remote_am_done_writing_to_top_neighbor[next_iter_comm_tile_flag_idx * nx +
                                                               next_iter_comm_tile_flag_idy] =
                            iter + 1;
                    }

                    cg::sync(cta);

                    if (cta.thread_rank() == 0) {
                        int cur_iter_comm_tile_flag_idx = comm_tile_idx + cur_iter_mod * num_flags;
                        int cur_iter_comm_tile_flag_idy = comm_tile_idy + cur_iter_mod * num_flags;
                        while (
                            local_is_bottom_neighbor_done_writing_to_me
                                [cur_iter_comm_tile_flag_idy * nx + cur_iter_comm_tile_flag_idx] !=
                            iter) {
                        }
                    }

                    cg::sync(cta);

                    if (iy < comm_tile_end_y && ix < comm_tile_end_x) {
                        const real last_row_val =
                            (1 / 6) * a[(iz_end - 1) * ny * nx + iy * nx + ix + 1] +
                            a[(iz_end - 1) * ny * nx + iy * nx + ix - 1] +
                            a[(iz_end - 1) * ny * nx + (iy + 1) * nx + ix] +
                            a[(iz_end - 1) * ny * nx + (iy - 1) * nx + ix] +
                            remote_my_halo_buffer_on_bottom_neighbor[cur_iter_mod * ny * nx +
                                                                     iy * nx + ix];

                        a_new[(iz_end - 1) * ny * nx + iy * nx + ix] = last_row_val;
                        local_halo_buffer_for_bottom_neighbor[next_iter_mod * ny * nx + iy * nx +
                                                              ix] = last_row_val;
                    }

                    cg::sync(cta);

                    if (cta.thread_rank() == 0) {
                        int next_iter_comm_tile_flag_idx =
                            comm_tile_idx + next_iter_mod * num_flags;
                        int next_iter_comm_tile_flag_idy =
                            comm_tile_idy + next_iter_mod * num_flags;

                        remote_am_done_writing_to_bottom_neighbor[next_iter_comm_tile_flag_idx *
                                                                      nx +
                                                                  next_iter_comm_tile_flag_idy] =
                            iter + 1;
                    }
                }
            }
        } else {
            for (int comp_tile_idx_z = 0; comp_tile_idx_z < num_comp_tiles_z; comp_tile_idx_z++) {
                int comp_tile_start_nz =
                    (comp_tile_idx_z == 0) ? iz_start + 1 : comp_tile_idx_z * comp_tile_size_z;

                int iz = base_iz + comp_tile_start_nz;

                int comp_tile_end_nz = (comp_tile_idx_z == (num_comp_tiles_z - 1))
                                           ? iz_end - 1
                                           : (comp_tile_idx_z + 1) * comp_tile_size_z;
                for (int comp_tile_idx_y = 0; comp_tile_idx_y < num_comp_tiles_y;
                     comp_tile_idx_y++) {
                    int comp_tile_start_ny =
                        (comp_tile_idx_y == 0) ? 1 : comp_tile_idx_y * comp_tile_size_y;

                    int iy = base_iy + comp_tile_start_ny;

                    int comp_tile_end_ny = (comp_tile_idx_y == (num_comp_tiles_y - 1))
                                               ? ny - 1
                                               : (comp_tile_idx_y + 1) * comp_tile_size_y;

                    for (int comp_tile_idx_x = 0; comp_tile_idx_x < num_comp_tiles_x;
                         comp_tile_idx_x++) {
                        int comp_tile_start_nx =
                            (comp_tile_idx_x == 0) ? 1 : comp_tile_idx_x * comp_tile_size_x;

                        int ix = base_ix + comp_tile_start_nx;

                        int comp_tile_end_nx = (comp_tile_idx_x == (num_comp_tiles_x - 1))
                                                   ? nx - 1
                                                   : (comp_tile_idx_x + 1) * comp_tile_size_x;

                        if (iz < comp_tile_end_nz && iy < comp_tile_end_ny &&
                            ix < comp_tile_end_nx) {
                            const real new_val = (1 / 6) * (a[iz * ny * nx + iy * nx + ix + 1] +
                                                            a[iz * ny * nx + iy * nx + ix - 1] +
                                                            a[iz * ny * nx + (iy + 1) * nx + ix] +
                                                            a[iz * ny * nx + (iy - 1) * nx + ix] +
                                                            a[(iz + 1) * ny * nx + iy * nx + ix] +
                                                            a[(iz - 1) * ny * nx + iy * nx + ix]);
                            a_new[iz * ny * nx + iy * nx + ix] = new_val;
                        }
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
}  // namespace SSMultiThreadedOneBlockComm

int SSMultiThreadedOneBlockComm::init(int argc, char *argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);
    const int nz = get_argval<int>(argv, argv + argc, "-nz", 16384);
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

        if (compare_to_single_gpu && 0 == dev_id) {
            CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * nz * sizeof(real)));
            CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * nz * sizeof(real)));

            runtime_serial_non_persistent = single_gpu(nz, ny, nx, iter_max, a_ref_h, 0, true);
        }

#pragma omp barrier

        int chunk_size;
        int chunk_size_low = (nz - 2) / num_devices;
        int chunk_size_high = chunk_size_low + 1;

        int height_per_gpu = nz / num_devices;

        cudaDeviceProp deviceProp{};
        CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, dev_id));
        int numSms = deviceProp.multiProcessorCount;

        constexpr int dim_block_x = 8;
        constexpr int dim_block_y = 8;
        constexpr int dim_block_z = 16;

        int comp_tile_size_x = 64;
        int comp_tile_size_y = 64;
        int comp_tile_size_z = 8;

        // int grid_dim_x = ((comp_tile_size_x + dim_block_x - 1) / dim_block_x);
        // int max_thread_blocks_y = (numSms - 1) / grid_dim_x;

        // comp_tile_size_y = dim_block_y * max_thread_blocks_y;

        // printf("Computation tile dimensions: %dx%dx%d\n", comp_tile_size_x,
        // comp_tile_size_y,comp_tile_size_z);

        int num_comp_tiles_x = nx / comp_tile_size_x + (nx % comp_tile_size_x != 0);
        int num_comp_tiles_y = ny / comp_tile_size_y + (ny % comp_tile_size_y != 0);
        int num_comp_tiles_z =
            height_per_gpu / comp_tile_size_z + (height_per_gpu % comp_tile_size_z != 0);

        int comm_tile_size_x = dim_block_x;
        int comm_tile_size_y = dim_block_y;
        int num_comm_tiles_x = nx / comm_tile_size_x + (nx % comm_tile_size_x != 0);
        int num_comm_tiles_y = ny / comm_tile_size_y + (ny % comm_tile_size_y != 0);
        int num_flags = 4 * num_comm_tiles_x * num_comm_tiles_y;

        // printf("Number of communication tiles: %d\n", num_comm_tiles);

        int num_ranks_low = num_devices * chunk_size_low + num_devices - (nz - 2);
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

        CUDA_RT_CALL(cudaMalloc(a + dev_id, nx * ny * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(a_new + dev_id, nx * ny * (chunk_size + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMemset(a[dev_id], 0, nx * ny * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(a_new[dev_id], 0, nx * (ny * chunk_size + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMalloc(halo_buffer_for_top_neighbor + dev_id, 2 * nx * ny * sizeof(real)));
        CUDA_RT_CALL(
            cudaMalloc(halo_buffer_for_bottom_neighbor + dev_id, 2 * nx * ny * sizeof(real)));

        CUDA_RT_CALL(
            cudaMemset(halo_buffer_for_top_neighbor[dev_id], 0, 2 * nx * ny * sizeof(real)));
        CUDA_RT_CALL(
            cudaMemset(halo_buffer_for_bottom_neighbor[dev_id], 0, 2 * nx * ny * sizeof(real)));

        CUDA_RT_CALL(cudaMalloc(is_top_done_computing_flags + dev_id, num_flags * sizeof(int)));
        CUDA_RT_CALL(cudaMalloc(is_bottom_done_computing_flags + dev_id, num_flags * sizeof(int)));

        CUDA_RT_CALL(cudaMemset(is_top_done_computing_flags[dev_id], 0, num_flags * sizeof(int)));
        CUDA_RT_CALL(
            cudaMemset(is_bottom_done_computing_flags[dev_id], 0, num_flags * sizeof(int)));

        // Calculate local domain boundaries
        int iz_start_global;  // My start index in the global array
        if (dev_id < num_ranks_low) {
            iz_start_global = dev_id * chunk_size_low + 1;
        } else {
            iz_start_global =
                num_ranks_low * chunk_size_low + (dev_id - num_ranks_low) * chunk_size_high + 1;
        }
        int iz_end_global = iz_start_global + chunk_size - 1;  // My last index in the global array

        int iz_start = 1;
        iz_end[dev_id] = (iz_end_global - iz_start_global + 1) + iz_start;

        // Set diriclet boundary conditions on left and right border
        initialize_boundaries<<<(ny / num_devices) / 128 + 1, 128>>>(
            a[dev_id], a_new[dev_id], PI, iz_start_global - 1, nx, ny, chunk_size + 2, ny);
        CUDA_RT_CALL(cudaGetLastError());

        CUDA_RT_CALL(cudaDeviceSynchronize());

        dim3 dim_grid(numSms, 1, 1);
        dim3 dim_block(dim_block_x, dim_block_y, dim_block_z);

        void *kernelArgs[] = {(void *)&a_new[dev_id],
                              (void *)&a[dev_id],
                              (void *)&iz_start,
                              (void *)&iz_end[dev_id],
                              (void *)&ny,
                              (void *)&nx,
                              (void *)&comp_tile_size_x,
                              (void *)&comp_tile_size_y,
                              (void *)&comp_tile_size_z,
                              (void *)&comm_tile_size_x,
                              (void *)&comm_tile_size_y,
                              (void *)&num_comp_tiles_x,
                              (void *)&num_comp_tiles_y,
                              (void *)&num_comp_tiles_z,
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
        if (iter_max % 2 == 1) {
            std::swap(a_new[dev_id], a[dev_id]);
        }

#pragma omp barrier
        double stop = omp_get_wtime();

        if (compare_to_single_gpu) {
            CUDA_RT_CALL(cudaMemcpy(
                a_h + iz_start_global * nx * ny, a[dev_id] + nx,
                std::min((ny - iz_start_global) * nx * ny, chunk_size * nx * ny) * sizeof(real),
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

        if (compare_to_single_gpu && 0 == dev_id) {
            CUDA_RT_CALL(cudaFreeHost(a_h));
            CUDA_RT_CALL(cudaFreeHost(a_ref_h));
        }
    }

    return 0;
}