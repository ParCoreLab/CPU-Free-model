/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This sample implements a conjugate gradient solver on multiple GPU using
 * Unified Memory optimized prefetching and usage hints.
 *
 */

// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <utility>

#include <omp.h>

#include "../../include/common.h"
#include "../../include/single-stream/pipelined.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace SingleStreamPipelined {

__device__ double grid_dot_result_delta = 0.0;
__device__ double grid_dot_result_gamma = 0.0;

__device__ void gpuSpMV(int *I, int *J, real *val, int nnz, int num_rows, real alpha,
                        real *inputVecX, real *outputVecY, const int num_allocated_tbs,
                        const int device_rank, const int num_devices, const PeerGroup &peer_group) {
    // This is kind of inelegant and could be better
    int local_subgrid_size = peer_group.calc_subgrid_size(num_allocated_tbs);
    int local_subgrid_rank = peer_group.calc_subgrid_thread_rank(num_allocated_tbs);

    int global_subgrid_size = local_subgrid_size * num_devices;
    int global_subgrid_rank = device_rank * local_subgrid_size + local_subgrid_rank;

    for (int i = global_subgrid_rank; i < num_rows; i += global_subgrid_size) {
        int row_elem = I[i];
        int next_row_elem = I[i + 1];
        int num_elems_this_row = next_row_elem - row_elem;

        real output = 0.0;
        for (int j = 0; j < num_elems_this_row; j++) {
            output += alpha * val[row_elem + j] * inputVecX[J[row_elem + j]];
        }

        outputVecY[i] = output;
    }
}

__device__ void gpuSaxpy(real *x, real *y, real a, int size, const PeerGroup &peer_group) {
    for (int i = peer_group.thread_rank(); i < size; i += peer_group.size()) {
        y[i] = a * x[i] + y[i];
    }
}

// Performs two dot products at the same time
// Used to perform <r, r> and <r, w> at the same time
// Can we combined the two atomicAdds somehow?
__device__ void gpuDotProductsMerged(real *vecA_delta, real *vecB_delta, real *vecA_gamma,
                                     real *vecB_gamma, int num_rows, const cg::thread_block &cta,
                                     const int device_rank, const int sMemSize,
                                     const PeerGroup &peer_group) {
    // First half (up to sMemSize / 2) will be used for delta
    // Second half (from sMemSize / 2) will be used for gamma
    extern __shared__ double tmp[];

    double *tmp_delta = (double *)tmp;
    double *tmp_gamma = (double *)&tmp_delta[sMemSize / (2 * sizeof(double))];

    double temp_sum_delta = 0.0;
    double temp_sum_gamma = 0.0;

    for (int i = peer_group.thread_rank(); i < num_rows; i += peer_group.size()) {
        temp_sum_delta += (double)(vecA_delta[i] * vecB_delta[i]);
        temp_sum_gamma += (double)(vecA_gamma[i] * vecB_gamma[i]);
    }

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    temp_sum_delta = cg::reduce(tile32, temp_sum_delta, cg::plus<double>());
    temp_sum_gamma = cg::reduce(tile32, temp_sum_gamma, cg::plus<double>());

    if (tile32.thread_rank() == 0) {
        tmp_delta[tile32.meta_group_rank()] = temp_sum_delta;
        tmp_gamma[tile32.meta_group_rank()] = temp_sum_gamma;
    }

    cg::sync(cta);

    if (tile32.meta_group_rank() == 0) {
        temp_sum_delta =
            tile32.thread_rank() < tile32.meta_group_size() ? tmp_delta[tile32.thread_rank()] : 0.0;
        temp_sum_delta = cg::reduce(tile32, temp_sum_delta, cg::plus<double>());

        temp_sum_gamma =
            tile32.thread_rank() < tile32.meta_group_size() ? tmp_gamma[tile32.thread_rank()] : 0.0;
        temp_sum_gamma = cg::reduce(tile32, temp_sum_gamma, cg::plus<double>());

        if (tile32.thread_rank() == 0) {
            atomicAdd(&grid_dot_result_delta, temp_sum_delta);
            atomicAdd(&grid_dot_result_gamma, temp_sum_gamma);
        }
    }
}

__device__ void gpuCopyVector(real *srcA, real *destB, int size, const PeerGroup &peer_group) {
    for (int i = peer_group.thread_rank(); i < size; i += peer_group.size()) {
        destB[i] = srcA[i];
    }
}

__device__ void gpuScaleVectorAndSaxpy(real *x, real *y, real a, real scale, int size,
                                       const PeerGroup &peer_group) {
    for (int i = peer_group.thread_rank(); i < size; i += peer_group.size()) {
        y[i] = a * x[i] + scale * y[i];
    }
}

__global__ void multiGpuConjugateGradient(int *I, int *J, real *val, real *x, real *r, real *p,
                                          real *s, real *z, real *w, real *q, real *ax0,
                                          double *dot_result_delta, double *dot_result_gamma,
                                          int nnz, int num_rows, real tol,
                                          MultiDeviceData multi_device_data, const int iter_max,
                                          const int sMemSize) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    PeerGroup peer_group(multi_device_data, grid);

    const int num_devices = multi_device_data.numDevices;
    const int device_rank = multi_device_data.deviceRank;

    real real_positive_one = 1.0;
    real real_negative_one = -1.0;

    real tmp_dot_delta_0 = 0.0;
    real tmp_dot_gamma_0 = 0.0;

    real tmp_dot_delta_1;
    real tmp_dot_gamma_1;

    real beta;
    real alpha;
    real negative_alpha;

    for (int i = peer_group.thread_rank(); i < num_rows; i += peer_group.size()) {
        r[i] = 1.0;
        x[i] = 0.0;
    }

    cg::sync(grid);

    // ax0 = AX0
    gpuSpMV(I, J, val, nnz, num_rows, real_positive_one, x, ax0, gridDim.x, device_rank,
            num_devices, peer_group);

    cg::sync(grid);

    // r0 = b0 - ax0
    // NOTE: b is a unit vector.
    gpuSaxpy(ax0, r, real_negative_one, num_rows, peer_group);

    cg::sync(grid);

    // w0 = Ar0
    gpuSpMV(I, J, val, nnz, num_rows, real_positive_one, r, w, gridDim.x, device_rank, num_devices,
            peer_group);

    cg::sync(grid);

    int k = 1;

    while (k <= iter_max) {
        if (peer_group.thread_rank() == 0) {
            *dot_result_delta = 0.0;
            *dot_result_gamma = 0.0;
        }

        peer_group.sync();

        gpuDotProductsMerged(r, r, r, w, num_rows, cta, device_rank, sMemSize, peer_group);

        cg::sync(grid);

        // Allocate one thread block for dot global reduction (`atomicAdd`s)
        // Rest are for SpMV
        if (blockIdx.x == gridDim.x - 1) {
            if (cta.thread_rank() == 0) {
                atomicAdd_system(dot_result_delta, grid_dot_result_delta);
                atomicAdd_system(dot_result_gamma, grid_dot_result_gamma);

                grid_dot_result_delta = 0.0;
                grid_dot_result_gamma = 0.0;
            }
        } else {
            gpuSpMV(I, J, val, nnz, num_rows, real_positive_one, w, q, gridDim.x - 1, device_rank,
                    num_devices, peer_group);
        }

        peer_group.sync();

        tmp_dot_delta_1 = *dot_result_delta;
        tmp_dot_gamma_1 = *dot_result_gamma;

        if (k > 1) {
            beta = tmp_dot_delta_1 / tmp_dot_delta_0;
            alpha = tmp_dot_delta_1 / (tmp_dot_gamma_1 - (beta / alpha) * tmp_dot_delta_1);
        } else {
            beta = 0.0;
            alpha = tmp_dot_delta_1 / tmp_dot_gamma_1;
        }

        // IMPORTANT: Is this peer sync necessary? Or would a grid sync suffice?
        peer_group.sync();

        // z_k = q_k + beta_k * z_(k-1)
        gpuScaleVectorAndSaxpy(q, z, real_positive_one, beta, num_rows, peer_group);

        // s_k = w_k + beta_k * s_(k-1)
        gpuScaleVectorAndSaxpy(w, s, real_positive_one, beta, num_rows, peer_group);

        // p_k = r_k = beta_k * p_(k-1)
        gpuScaleVectorAndSaxpy(r, p, real_positive_one, beta, num_rows, peer_group);

        cg::sync(grid);

        // x_(k+1) = x_k + alpha_k * p_k
        gpuSaxpy(p, x, alpha, num_rows, peer_group);

        negative_alpha = -alpha;

        // r_(k+1) = r_k - alpha_k * s_k
        gpuSaxpy(s, r, negative_alpha, num_rows, peer_group);

        // w_(k+1) = w_k - alpha_k * z_k
        gpuSaxpy(z, w, negative_alpha, num_rows, peer_group);

        tmp_dot_delta_0 = (real)tmp_dot_delta_1;
        tmp_dot_gamma_0 = (real)tmp_dot_gamma_1;

        peer_group.sync();

        k++;
    }
}
}  // namespace SingleStreamPipelined

int SingleStreamPipelined::init(int argc, char *argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 10000);
    std::string matrix_path_str = get_argval<std::string>(argv, argv + argc, "-matrix_path", "");
    const bool compare_to_single_gpu = get_arg(argv, argv + argc, "-compare-single-gpu");
    const bool compare_to_cpu = get_arg(argv, argv + argc, "-compare-cpu");

    char *matrix_path_char = const_cast<char *>(matrix_path_str.c_str());
    bool generate_random_tridiag_matrix = matrix_path_str.empty();

    std::string matrix_name = std::filesystem::path(matrix_path_str).stem();

    if (generate_random_tridiag_matrix) {
        matrix_name = "random tridiagonal";
    }

    // std::cout << "Running on matrix: " << matrix_name << "\n" << std::endl;

    int num_devices = 0;
    double single_gpu_runtime;

    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

    int num_rows = 0;
    int num_cols = 0;
    int nnz = 0;

    // Structure used for cross-grid synchronization.
    MultiDeviceData multi_device_data;

    int *host_I = NULL;
    int *host_J = NULL;
    real *host_val = NULL;

    real *x_ref_single_gpu = NULL;

    real *s_cpu = NULL;
    real *r_cpu = NULL;
    real *p_cpu = NULL;
    real *x_ref_cpu = NULL;

    int *um_I = NULL;
    int *um_J = NULL;
    real *um_val = NULL;

    real r1;

    real *um_x;
    real *um_r;
    real *um_p;
    real *um_s;
    real *um_z;
    real *um_w;
    real *um_q;
    real *um_ax0;

    double *dot_result_delta;
    double *dot_result_gamma;

#pragma omp parallel num_threads(num_devices)
    {
        int gpu_idx = omp_get_thread_num();

        CUDA_RT_CALL(cudaSetDevice(gpu_idx));

        cudaStream_t mainStream;
        CUDA_RT_CALL(cudaStreamCreate(&mainStream));

#pragma omp barrier

        for (int gpu_idx_j = 0; gpu_idx_j < num_devices; gpu_idx_j++) {
            if (gpu_idx != gpu_idx_j) {
                CUDA_RT_CALL(cudaDeviceEnablePeerAccess(gpu_idx_j, 0));
            }
        }

#pragma omp barrier

#pragma omp master
        {
            if (generate_random_tridiag_matrix) {
                num_rows = 10485760 * 2;
                num_cols = num_rows;

                nnz = (num_rows - 2) * 3 + 4;

                CUDA_RT_CALL(cudaMallocManaged((void **)&um_I, sizeof(int) * (num_rows + 1)));
                CUDA_RT_CALL(cudaMallocManaged((void **)&um_J, sizeof(int) * nnz));
                CUDA_RT_CALL(cudaMallocManaged((void **)&um_val, sizeof(real) * nnz));

                host_val = (real *)malloc(sizeof(real) * nnz);

                /* Generate a random tridiagonal symmetric matrix in CSR format */
                genTridiag(um_I, um_J, host_val, num_rows, nnz);

                memcpy(um_val, host_val, sizeof(real) * nnz);

            } else {
                if (loadMMSparseMatrix<real>(matrix_path_char, 'd', true, &num_rows, &num_cols,
                                             &nnz, &host_val, &host_I, &host_J, true)) {
                    exit(EXIT_FAILURE);
                }

                CUDA_RT_CALL(cudaMallocManaged((void **)&um_I, sizeof(int) * (num_rows + 1)));
                CUDA_RT_CALL(cudaMallocManaged((void **)&um_J, sizeof(int) * nnz));
                CUDA_RT_CALL(cudaMallocManaged((void **)&um_val, sizeof(real) * nnz));

                memcpy(um_I, host_I, sizeof(int) * (num_rows + 1));
                memcpy(um_J, host_J, sizeof(int) * nnz);
                memcpy(um_val, host_val, sizeof(real) * nnz);
            }
        }

#pragma omp barrier

#pragma omp master
        {
            if (compare_to_single_gpu) {
                CUDA_RT_CALL(cudaMallocHost(&x_ref_single_gpu, num_rows * sizeof(real)));

                // single_gpu_runtime = SingleGPUStandardDiscrete::run_single_gpu(
                //     iter_max, um_I, um_J, um_val, x_ref_single_gpu, num_rows, nnz);

                single_gpu_runtime = SingleGPUPipelinedDiscrete::run_single_gpu(
                    iter_max, um_I, um_J, um_val, x_ref_single_gpu, num_rows, nnz);
            }

            if (compare_to_cpu) {
                s_cpu = (real *)malloc(sizeof(real) * num_rows);
                r_cpu = (real *)malloc(sizeof(real) * num_rows);
                p_cpu = (real *)malloc(sizeof(real) * num_rows);

                CUDA_RT_CALL(cudaMallocHost(&x_ref_cpu, num_rows * sizeof(real)));

                for (int i = 0; i < num_rows; i++) {
                    r_cpu[i] = 1.0;
                    s_cpu[i] = 0.0;
                    x_ref_cpu[i] = 0.0;
                }

                CPU::cpuConjugateGrad(iter_max, um_I, um_J, um_val, x_ref_cpu, s_cpu, p_cpu, r_cpu,
                                      nnz, num_rows, tol);
            }

            CUDA_RT_CALL(cudaDeviceSynchronize());
        }

#pragma omp barrier

#pragma omp master
        {
            CUDA_RT_CALL(cudaMallocManaged((void **)&um_x, sizeof(real) * num_rows));

            CUDA_RT_CALL(cudaMallocManaged((void **)&dot_result_delta, sizeof(double)));
            CUDA_RT_CALL(cudaMallocManaged((void **)&dot_result_gamma, sizeof(double)));

            CUDA_RT_CALL(cudaMemset(dot_result_delta, 0, sizeof(double)));
            CUDA_RT_CALL(cudaMemset(dot_result_gamma, 0, sizeof(double)));

            // temp memory for ConjugateGradient
            CUDA_RT_CALL(cudaMallocManaged((void **)&um_r, num_rows * sizeof(real)));
            CUDA_RT_CALL(cudaMallocManaged((void **)&um_p, num_rows * sizeof(real)));
            CUDA_RT_CALL(cudaMallocManaged((void **)&um_s, num_rows * sizeof(real)));
            CUDA_RT_CALL(cudaMallocManaged((void **)&um_z, num_rows * sizeof(real)));
            CUDA_RT_CALL(cudaMallocManaged((void **)&um_w, num_rows * sizeof(real)));
            CUDA_RT_CALL(cudaMallocManaged((void **)&um_q, num_rows * sizeof(real)));
            CUDA_RT_CALL(cudaMallocManaged((void **)&um_ax0, num_rows * sizeof(real)));
        }

        int sMemSize = 2 * (sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1));
        int numBlocksPerSm = INT_MAX;
        int numThreads = THREADS_PER_BLOCK;

#pragma omp barrier

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        int numSms = deviceProp.multiProcessorCount;

        CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocksPerSm, multiGpuConjugateGradient, numThreads, sMemSize));

        if (!numBlocksPerSm) {
            printf("Max active blocks per SM is returned as 0. Exiting!\n");

            exit(EXIT_FAILURE);
        }

#pragma omp barrier

        dim3 dimGrid(numSms * numBlocksPerSm, 1, 1), dimBlock(numThreads, 1, 1);

#pragma omp master
        {
            CUDA_RT_CALL(
                cudaHostAlloc(&multi_device_data.hostMemoryArrivedList,
                              (num_devices - 1) * sizeof(*multi_device_data.hostMemoryArrivedList),
                              cudaHostAllocPortable));
            memset(multi_device_data.hostMemoryArrivedList, 0,
                   (num_devices - 1) * sizeof(*multi_device_data.hostMemoryArrivedList));
            multi_device_data.numDevices = num_devices;
            multi_device_data.deviceRank = 0;
        }

        void *kernelArgs[] = {
            (void *)&um_I,
            (void *)&um_J,
            (void *)&um_val,
            (void *)&um_x,
            (void *)&um_r,
            (void *)&um_p,
            (void *)&um_s,
            (void *)&um_z,
            (void *)&um_w,
            (void *)&um_q,
            (void *)&um_ax0,
            (void *)&dot_result_delta,
            (void *)&dot_result_gamma,
            (void *)&nnz,
            (void *)&num_rows,
            (void *)&tol,
            (void *)&multi_device_data,
            (void *)&iter_max,
            (void *)&sMemSize,
        };

        CUDA_RT_CALL(cudaDeviceSynchronize());

#pragma omp barrier

        double start = omp_get_wtime();

#pragma omp critical
        {
            multi_device_data.deviceRank = gpu_idx;
            CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *)multiGpuConjugateGradient, dimGrid,
                                                     dimBlock, kernelArgs, sMemSize, mainStream));
        }

        CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

#pragma omp barrier

        double stop = omp_get_wtime();

#pragma omp master
        {
            report_results(num_rows, x_ref_single_gpu, x_ref_cpu, um_x, num_devices,
                           single_gpu_runtime, start, stop, compare_to_single_gpu, compare_to_cpu);
        }

#pragma omp barrier

#pragma omp master
        {
            CUDA_RT_CALL(cudaFreeHost(multi_device_data.hostMemoryArrivedList));
            CUDA_RT_CALL(cudaFree(um_I));
            CUDA_RT_CALL(cudaFree(um_J));
            CUDA_RT_CALL(cudaFree(um_val));
            CUDA_RT_CALL(cudaFree(um_x));
            CUDA_RT_CALL(cudaFree(um_r));
            CUDA_RT_CALL(cudaFree(um_p));
            CUDA_RT_CALL(cudaFree(um_s));
            CUDA_RT_CALL(cudaFree(dot_result_delta));
            CUDA_RT_CALL(cudaFree(dot_result_gamma));
            free(host_val);

            CUDA_RT_CALL(cudaFreeHost(x_ref_single_gpu));
        }

        CUDA_RT_CALL(cudaStreamDestroy(mainStream));
    }

    return 0;
}
