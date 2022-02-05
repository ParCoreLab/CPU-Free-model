/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */
#include <cmath>
#include <cstdio>
#include <iostream>

#include <omp.h>

#include <cooperative_groups.h>

#include "../../include/common.h"
#include "../../include/single-stream/single-threaded.cuh"

namespace cg = cooperative_groups;

namespace SSSingleThreaded {
__global__ void jacobi_kernel(real* a_new, real* a, const int iy_start, const int iy_end,
                              const int nx, real* a_new_top, real* a_top, const int top_iy,
                              real* a_new_bottom, real* a_bottom, const int bottom_iy,
                              const int iter_max,
                              volatile int* local_is_top_neighbor_done_writing_to_me,
                              volatile int* local_is_bottom_neighbor_done_writing_to_me,
                              volatile int* remote_am_done_writing_to_top_neighbor,
                              volatile int* remote_am_done_writing_to_bottom_neighbor) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    unsigned int grid_dim_x = (nx + blockDim.x - 1) / blockDim.x;
    unsigned int block_idx_y = blockIdx.x / grid_dim_x;
    unsigned int block_idx_x = blockIdx.x % grid_dim_x;

    unsigned int iy = block_idx_y * blockDim.y + threadIdx.y + iy_start;
    unsigned int ix = block_idx_x * blockDim.x + threadIdx.x + 1;

    //    real local_l2_norm = 0.0;
    int iter = 0;

    int cur_iter_mod = 0;
    int next_iter_mod = 1;
    int temp_iter_mod = 0;

    while (iter < iter_max) {
        //    One thread block does communication (and a bit of computation)
        if (blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y - 1) {
            unsigned int col = threadIdx.y * blockDim.x + threadIdx.x + 1;

            if (col < nx - 1) {
                // Wait until top GPU puts its bottom row as my top halo
                while (local_is_top_neighbor_done_writing_to_me[cur_iter_mod] != iter) {
                }

                const real first_row_val =
                    0.25 * (a[iy_start * nx + col + 1] + a[iy_start * nx + col - 1] +
                            a[(iy_start + 1) * nx + col] + a[(iy_start - 1) * nx + col]);
                a_new[iy_start * nx + col] = first_row_val;

                while (local_is_bottom_neighbor_done_writing_to_me[cur_iter_mod] != iter) {
                }

                const real last_row_val =
                    0.25 * (a[(iy_end - 1) * nx + col + 1] + a[(iy_end - 1) * nx + col - 1] +
                            a[(iy_end - 2) * nx + col] + a[(iy_end)*nx + col]);
                a_new[(iy_end - 1) * nx + col] = last_row_val;

                //                if (calculate_norm) {
                //                    real first_row_residue = first_row_val - a[iy_start * nx +
                //                    col]; real last_row_residue = last_row_val - a[iy_end * nx +
                //                    col];
                //
                //                    local_l2_norm += first_row_residue * first_row_residue;
                //                    local_l2_norm += last_row_residue * last_row_residue;
                //                }

                // Communication
                a_new_top[top_iy * nx + col] = first_row_val;
                a_new_bottom[bottom_iy * nx + col] = last_row_val;
            }

            cg::sync(cta);

            if (threadIdx.x == 0 && threadIdx.y == 0) {
                remote_am_done_writing_to_top_neighbor[next_iter_mod] = iter + 1;
                remote_am_done_writing_to_bottom_neighbor[next_iter_mod] = iter + 1;
            }
        } else if (iy > iy_start && iy < (iy_end - 1) && ix < (nx - 1)) {
            const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                         a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
            a_new[iy * nx + ix] = new_val;

            //            if (calculate_norm) {
            //                real residue = new_val - a[iy * nx + ix];
            //                local_l2_norm += residue * residue;
            //            }
        }

        real* temp_pointer_first = a_new;
        a_new = a;
        a = temp_pointer_first;

        real* temp_pointer_second = a_new_top;
        a_new_top = a_top;
        a_top = temp_pointer_second;

        real* temp_pointer_third = a_new_bottom;
        a_new_bottom = a_bottom;
        a_bottom = temp_pointer_third;

        iter++;

        temp_iter_mod = cur_iter_mod;
        cur_iter_mod = next_iter_mod;
        next_iter_mod = temp_iter_mod;

        cg::sync(grid);
    }
}
}  // namespace SSSingleThreaded

int SSSingleThreaded::init(int argc, char* argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);

    real* a[MAX_NUM_DEVICES];
    real* a_new[MAX_NUM_DEVICES];

    int iy_start[MAX_NUM_DEVICES];
    int iy_end[MAX_NUM_DEVICES];
    int chunk_size[MAX_NUM_DEVICES];

    int iy_start_bottom = 0;

    real* a_ref_h;
    real* a_h;
    double runtime_serial = 0.0;

    int* is_top_done_computing_flags[MAX_NUM_DEVICES];
    int* is_bottom_done_computing_flags[MAX_NUM_DEVICES];

    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));
    //    real l2_norm = 1.0;

    for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaFree(0));

        if (0 == dev_id) {
            CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * sizeof(real)));
            CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * sizeof(real)));
            runtime_serial = single_gpu(nx, ny, iter_max, a_ref_h, 0, true);
        }

        // ny - 2 rows are distributed amongst `size` ranks in such a way
        // that each rank gets either (ny - 2) / size or (ny - 2) / size + 1 rows.
        // This optimizes load balancing when (ny - 2) % size != 0
        int chunk_size_low = (ny - 2) / num_devices;
        int chunk_size_high = chunk_size_low + 1;
        // To calculate the number of ranks that need to compute an extra row,
        // the following formula is derived from this equation:
        // num_ranks_low * chunk_size_low + (size - num_ranks_low) * (chunk_size_low + 1) = ny - 2
        int num_ranks_low = num_devices * chunk_size_low + num_devices -
                            (ny - 2);  // Number of ranks with chunk_size = chunk_size_low
        if (dev_id < num_ranks_low)
            chunk_size[dev_id] = chunk_size_low;
        else
            chunk_size[dev_id] = chunk_size_high;

        CUDA_RT_CALL(cudaMalloc(a + dev_id, nx * (chunk_size[dev_id] + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(a_new + dev_id, nx * (chunk_size[dev_id] + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMemset(a[dev_id], 0, nx * (chunk_size[dev_id] + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(a_new[dev_id], 0, nx * (chunk_size[dev_id] + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMalloc(is_top_done_computing_flags + dev_id, 2 * sizeof(int)));
        CUDA_RT_CALL(cudaMalloc(is_bottom_done_computing_flags + dev_id, 2 * sizeof(int)));

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

        iy_start[dev_id] = 1;
        iy_end[dev_id] = iy_start[dev_id] + chunk_size[dev_id];

        // Set diriclet boundary conditions on left and right boarder
        initialize_boundaries<<<(ny / num_devices) / 128 + 1, 128>>>(
            a[dev_id], a_new[dev_id], PI, iy_start_global - 1, nx, (chunk_size[dev_id] + 2), ny);
        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());

        const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);
        int canAccessPeer = 0;
        CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, top));
        if (canAccessPeer) {
            CUDA_RT_CALL(cudaDeviceEnablePeerAccess(top, 0));
        }
        const int bottom = (dev_id + 1) % num_devices;
        if (top != bottom) {
            canAccessPeer = 0;
            CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, bottom));
            if (canAccessPeer) {
                CUDA_RT_CALL(cudaDeviceEnablePeerAccess(bottom, 0));
            }
        }

        CUDA_RT_CALL(cudaDeviceSynchronize());
    }

    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 32;
    constexpr int num_threads = 1024;

    cudaDeviceProp deviceProp{};
    CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, dev_id));
    int numSms = deviceProp.multiProcessorCount;

    dim3 dim_grid(numSms, 1, 1);
    dim3 dim_block(dim_block_x, dim_block_y);

    for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaDeviceSynchronize());
    }

    double start = omp_get_wtime();

    for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
        const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);
        const int bottom = (dev_id + 1) % num_devices;
        CUDA_RT_CALL(cudaSetDevice(dev_id));



        void* kernelArgs[] = {(void*)&a_new[dev_id],
                              (void*)&a[dev_id],
                              (void*)&iy_start,
                              (void*)&iy_end[dev_id],
                              (void*)&nx,
                              (void*)&a_new[top],
                              (void*)&a[top],
                              (void*)&iy_end[top],
                              (void*)&a_new[bottom],
                              (void*)&a[bottom],
                              (void*)&iy_start_bottom,
                              (void*)&iter_max,
                              (void*)&is_top_done_computing_flags[dev_id],
                              (void*)&is_bottom_done_computing_flags[dev_id],
                              (void*)&is_bottom_done_computing_flags[top],
                              (void*)&is_top_done_computing_flags[bottom]};

        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void*)SSSingleThreaded::jacobi_kernel, dim_grid,
                                                 dim_block, kernelArgs, 0, nullptr));
    }

    for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaDeviceSynchronize());
    }

    double stop = omp_get_wtime();

    int offset = nx;
    for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
        CUDA_RT_CALL(
            cudaMemcpy(a_h + offset, a[dev_id] + nx,
                       std::min((nx * ny) - offset, nx * chunk_size[dev_id]) * sizeof(real),
                       cudaMemcpyDeviceToHost));
        offset += std::min(chunk_size[dev_id] * nx, (nx * ny) - offset);
    }

    bool result_correct = true;
    for (int iy = 1; result_correct && (iy < (ny - 1)); ++iy) {
        for (int ix = 1; result_correct && (ix < (nx - 1)); ++ix) {
            if (std::fabs(a_ref_h[iy * nx + ix] - a_h[iy * nx + ix]) > tol) {
                fprintf(stderr,
                        "ERROR: a[%d * %d + %d] = %f does not match %f "
                        "(reference)\n",
                        iy, nx, ix, a_h[iy * nx + ix], a_ref_h[iy * nx + ix]);
                result_correct = false;
            }
        }
    }

    if (result_correct) {
        printf("Num GPUs: %d.\n", num_devices);
        printf(
            "%dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: %8.2f, "
            "efficiency: %8.2f \n",
            ny, nx, runtime_serial, num_devices, (stop - start), runtime_serial / (stop - start),
            runtime_serial / (num_devices * (stop - start)) * 100);
    }
}