/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */
#include <cmath>
#include <cstdio>
#include <iostream>

#include <omp.h>

#include <cooperative_groups.h>

#include "../../include/common.h"
#include "../../include/single-stream/multi-threaded.cuh"

namespace cg = cooperative_groups;

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

__global__ void jacobi_kernel_single_gpu(real* __restrict__ const a_new,
                                         const real* __restrict__ const a,
                                         real* __restrict__ const l2_norm, const int iy_start,
                                         const int iy_end, const int nx,
                                         const bool calculate_norm) {
    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    real local_l2_norm = 0.0;

    if (iy < iy_end && ix < (nx - 1)) {
        const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                     a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
        a_new[iy * nx + ix] = new_val;
        if (calculate_norm) {
            real residue = new_val - a[iy * nx + ix];
            local_l2_norm += residue * residue;
        }
    }
    if (calculate_norm) {
        atomicAdd(l2_norm, local_l2_norm);
    }
}

__global__ void jacobi_kernel(real* a_new, const real* a, const int iy_start, const int iy_end,
                              const int nx, real* a_new_top, const int top_iy, real* a_new_bottom,
                              const int bottom_iy, const int iter_max,
                              volatile int* local_is_top_neighbor_done_writing_to_me,
                              volatile int* local_is_bottom_neighbor_done_writing_to_me,
                              volatile int* remote_am_done_writing_to_top_neighbor,
                              volatile int* remote_am_done_writing_to_bottom_neighbor,
                              const bool calculate_norm) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;

    real local_l2_norm = 0.0;
    int iter = 0;

    while (iter < iter_max) {
        //    One thread block does communication (and a bit of computation)
        if (blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y - 1) {
            unsigned int iy = threadIdx.y + iy_start;
            unsigned int ix = threadIdx.x + 1;
            unsigned int col = iy * blockDim.x + ix;

            if (col < nx) {
                // Wait until top GPU puts its bottom row as my top halo
                while (local_is_top_neighbor_done_writing_to_me[(iter % 2)] != iter) {
                }

                const real first_row_val =
                    0.25 * (a[iy_start * nx + col + 1] + a[iy_start * nx + col - 1] +
                            a[(iy_start + 1) * nx + col] + a[(iy_start - 1) * nx + col]);

                while (local_is_bottom_neighbor_done_writing_to_me[(iter % 2)] != iter) {
                }

                const real last_row_val =
                    0.25 * (a[(iy_end - 1) * nx + col + 1] + a[(iy_end - 1) * nx + col - 1] +
                            a[(iy_end - 2) * nx + col] + a[(iy_end)*nx + col]);

                if (calculate_norm) {
                    real first_row_residue = first_row_val - a[iy_start * nx + col];
                    real last_row_residue = last_row_val - a[iy_end * nx + col];

                    local_l2_norm += first_row_residue * first_row_residue;
                    local_l2_norm += last_row_residue * last_row_residue;
                }

                // Communication
                a_new_top[top_iy * nx + col] = first_row_val;
                a_new_bottom[bottom_iy * nx + col] = last_row_val;
            }

            cg::sync(cta);

            if (threadIdx.x == 0 && threadIdx.y == 0) {
                remote_am_done_writing_to_top_neighbor[(iter + 1) % 2] = iter + 1;
                remote_am_done_writing_to_bottom_neighbor[(iter + 1) % 2] = iter + 1;
            }
        } else if (iy > iy_start && iy < iy_end - 1 && ix < (nx - 1)) {
            const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                         a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
            a_new[iy * nx + ix] = new_val;

            if (calculate_norm) {
                real residue = new_val - a[iy * nx + ix];
                local_l2_norm += residue * residue;
            }
        }

        real* temp_pointer = a_new;
        a = a_new;
        a_new = temp_pointer;

        iter++;

        cg::sync(grid);
    }
}

int SSMultiThreaded::init(int argc, char* argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);

    real* a[MAX_NUM_DEVICES];
    real* a_new[MAX_NUM_DEVICES];
    int iy_end[MAX_NUM_DEVICES];

    real* a_ref_h;
    real* a_h;
    double runtime_serial = 0.0;

    int* is_top_done_computing_flags[MAX_NUM_DEVICES];
    int* is_bottom_done_computing_flags[MAX_NUM_DEVICES];

    bool result_correct = true;
    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));
    real l2_norm = 1.0;

    // Getting device properties and calculating block dimensions
    // Maybe put a warning if not all gpus have the same sm count
    cudaDeviceProp deviceProp{};
    CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, 0));
    int numSms = deviceProp.multiProcessorCount;

    constexpr int THREADS_PER_BLOCK = 256;

    int numBlocksPerSm = 0;
    int numThreads = THREADS_PER_BLOCK;

    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, jacobi_kernel,
                                                               numThreads, 0));

    int blocks_each = (int)sqrt(numSms * numBlocksPerSm);
    int threads_each = (int)sqrt(THREADS_PER_BLOCK);
    dim3 dimGrid(blocks_each, blocks_each), dimBlock(threads_each, threads_each);

#pragma omp parallel num_threads(num_devices) shared(l2_norm)
    {
        real* l2_norm_d;
        real* l2_norm_h;

        int dev_id = omp_get_thread_num();

        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaFree(nullptr));

        if (0 == dev_id) {
            CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * sizeof(real)));
            CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * sizeof(real)));

            // Passing 0 for nccheck for now
            runtime_serial = single_gpu(nx, ny, iter_max, a_ref_h, 0, true);
        }

#pragma omp barrier

        int chunk_size;
        int chunk_size_low = (ny - 2) / num_devices;
        int chunk_size_high = chunk_size_low + 1;

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
        int iy_end_global = iy_start_global + chunk_size - 1;  // My last index in the global array

        int iy_start = 1;
        iy_end[dev_id] = (iy_end_global - iy_start_global + 1) + iy_start;
        int iy_start_bottom = 0;

        // Set diriclet boundary conditions on left and right border
        initialize_boundaries<<<(ny / num_devices) / 128 + 1, 128>>>(
            a[dev_id], a_new[dev_id], PI, iy_start_global - 1, nx, (chunk_size + 2), ny);
        CUDA_RT_CALL(cudaGetLastError());

        CUDA_RT_CALL(cudaDeviceSynchronize());

        constexpr int dim_block_x = 16;
        constexpr int dim_block_y = 16;

        void* kernelArgs[] = {(void*)&a_new[dev_id],
                              (void*)&a[dev_id],
                              (void*)&iy_start,
                              (void*)&iy_end[dev_id],
                              (void*)&nx,
                              (void*)&a_new[top],
                              (void*)&iy_end[top],
                              (void*)&a_new[bottom],
                              (void*)&iy_start_bottom,
                              (void*)&iter_max,
                              (void*)&is_top_done_computing_flags[dev_id],
                              (void*)&is_bottom_done_computing_flags[dev_id],
                              (void*)&is_bottom_done_computing_flags[top],
                              (void*)&is_top_done_computing_flags[bottom]};

#pragma omp barrier
        double start = omp_get_wtime();

        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void*)jacobi_kernel, dimGrid, dimBlock,
                                                 kernelArgs, 0, nullptr));

        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());

        double stop = omp_get_wtime();

        CUDA_RT_CALL(
            cudaMemcpy(a_h + iy_start_global * nx, a[dev_id] + nx,
                       std::min((ny - iy_start_global) * nx, chunk_size * nx) * sizeof(real),
                       cudaMemcpyDeviceToHost));
#pragma omp barrier

#pragma omp master
        {
            result_correct = true;
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
                    "%dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: "
                    "%8.2f, "
                    "efficiency: %8.2f \n",
                    ny, nx, runtime_serial, num_devices, (stop - start),
                    runtime_serial / (stop - start),
                    runtime_serial / (num_devices * (stop - start)) * 100);
            }
        }
    }
}

double single_gpu(const int nx, const int ny, const int iter_max, real* const a_ref_h,
                  const int nccheck, const bool print) {
    real* a;
    real* a_new;

    cudaStream_t compute_stream;
    cudaStream_t push_top_stream;
    cudaStream_t push_bottom_stream;
    cudaEvent_t compute_done;
    cudaEvent_t push_top_done;
    cudaEvent_t push_bottom_done;

    real* l2_norm_d;
    real* l2_norm_h;

    int iy_start = 1;
    int iy_end = (ny - 1);

    CUDA_RT_CALL(cudaMalloc(&a, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc(&a_new, nx * ny * sizeof(real)));

    CUDA_RT_CALL(cudaMemset(a, 0, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * ny * sizeof(real)));

    // Set diriclet boundary conditions on left and right boarder
    initialize_boundaries<<<ny / 128 + 1, 128>>>(a, a_new, PI, 0, nx, ny, ny);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());

    CUDA_RT_CALL(cudaStreamCreate(&compute_stream));
    CUDA_RT_CALL(cudaStreamCreate(&push_top_stream));
    CUDA_RT_CALL(cudaStreamCreate(&push_bottom_stream));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&compute_done, cudaEventDisableTiming));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&push_top_done, cudaEventDisableTiming));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&push_bottom_done, cudaEventDisableTiming));

    CUDA_RT_CALL(cudaMalloc(&l2_norm_d, sizeof(real)));
    CUDA_RT_CALL(cudaMallocHost(&l2_norm_h, sizeof(real)));

    CUDA_RT_CALL(cudaDeviceSynchronize());

    if (print)
        printf(
            "Single GPU jacobi relaxation: %d iterations on %d x %d mesh with "
            "norm "
            "check every %d iterations\n",
            iter_max, ny, nx, nccheck);

    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 32;
    dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x, (ny + dim_block_y - 1) / dim_block_y, 1);

    int iter = 0;
    bool calculate_norm = true;
    real l2_norm = 1.0;

    double start = omp_get_wtime();
    PUSH_RANGE("Jacobi solve", 0)
    while (l2_norm > tol && iter < iter_max) {
        CUDA_RT_CALL(cudaMemsetAsync(l2_norm_d, 0, sizeof(real), compute_stream));

        CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_top_done, 0));
        CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_bottom_done, 0));

        calculate_norm = (iter % nccheck) == 0 || (print && ((iter % 100) == 0));
        jacobi_kernel_single_gpu<<<dim_grid, {dim_block_x, dim_block_y, 1}, 0, compute_stream>>>(
            a_new, a, l2_norm_d, iy_start, iy_end, nx, calculate_norm);
        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaEventRecord(compute_done, compute_stream));

        if (calculate_norm) {
            CUDA_RT_CALL(cudaMemcpyAsync(l2_norm_h, l2_norm_d, sizeof(real), cudaMemcpyDeviceToHost,
                                         compute_stream));
        }

        // Apply periodic boundary conditions

        CUDA_RT_CALL(cudaStreamWaitEvent(push_top_stream, compute_done, 0));
        CUDA_RT_CALL(cudaMemcpyAsync(a_new, a_new + (iy_end - 1) * nx, nx * sizeof(real),
                                     cudaMemcpyDeviceToDevice, push_top_stream));
        CUDA_RT_CALL(cudaEventRecord(push_top_done, push_top_stream));

        CUDA_RT_CALL(cudaStreamWaitEvent(push_bottom_stream, compute_done, 0));
        CUDA_RT_CALL(cudaMemcpyAsync(a_new + iy_end * nx, a_new + iy_start * nx, nx * sizeof(real),
                                     cudaMemcpyDeviceToDevice, compute_stream));
        CUDA_RT_CALL(cudaEventRecord(push_bottom_done, push_bottom_stream));

        if (calculate_norm) {
            CUDA_RT_CALL(cudaStreamSynchronize(compute_stream));
            l2_norm = *l2_norm_h;
            l2_norm = std::sqrt(l2_norm);
            if (print && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, l2_norm);
        }

        std::swap(a_new, a);
        iter++;
    }
    POP_RANGE
    double stop = omp_get_wtime();

    CUDA_RT_CALL(cudaMemcpy(a_ref_h, a, nx * ny * sizeof(real), cudaMemcpyDeviceToHost));

    CUDA_RT_CALL(cudaEventDestroy(push_bottom_done));
    CUDA_RT_CALL(cudaEventDestroy(push_top_done));
    CUDA_RT_CALL(cudaEventDestroy(compute_done));
    CUDA_RT_CALL(cudaStreamDestroy(push_bottom_stream));
    CUDA_RT_CALL(cudaStreamDestroy(push_top_stream));
    CUDA_RT_CALL(cudaStreamDestroy(compute_stream));

    CUDA_RT_CALL(cudaFreeHost(l2_norm_h));
    CUDA_RT_CALL(cudaFree(l2_norm_d));

    CUDA_RT_CALL(cudaFree(a_new));
    CUDA_RT_CALL(cudaFree(a));
    return (stop - start);
}