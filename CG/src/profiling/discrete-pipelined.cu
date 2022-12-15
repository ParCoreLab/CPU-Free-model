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
#include "../../include/profiling/discrete-pipelined.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace ProfilingDiscretePipelined {

// delta => <r, r>
// gamma => <r, w>
__device__ double grid_dot_result_delta = 0.0;
__device__ double grid_dot_result_gamma = 0.0;

// Performs two dot products at the same time
// Used to perform <r, r> and <r, w> at the same time
// Can we combined the two atomicAdds somehow?

__global__ void gpuDotProductsMerged(real *vecA_delta, real *vecB_delta, real *vecA_gamma,
                                     real *vecB_gamma, int num_rows, const int device_rank,
                                     const int num_devices, const int sMemSize) {
    cg::thread_block cta = cg::this_thread_block();

    size_t local_grid_size = gridDim.x * blockDim.x;
    size_t local_grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    size_t global_grid_size = local_grid_size * num_devices;
    size_t global_grid_rank = device_rank * local_grid_size + local_grid_rank;

    extern __shared__ double tmp[];

    double *tmp_delta = (double *)tmp;
    double *tmp_gamma = (double *)&tmp_delta[sMemSize / (2 * sizeof(double))];

    double temp_sum_delta = 0.0;
    double temp_sum_gamma = 0.0;

    for (size_t i = global_grid_rank; i < num_rows; i += global_grid_size) {
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

__global__ void addLocalDotContributions(double *dot_result_delta, double *dot_result_gamma) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        atomicAdd_system(dot_result_delta, grid_dot_result_delta);
        atomicAdd_system(dot_result_gamma, grid_dot_result_gamma);

        grid_dot_result_delta = 0.0;
        grid_dot_result_gamma = 0.0;
    }
}

__global__ void resetLocalDotProducts(double *dot_result_delta, double *dot_result_gamma,
                                      const int gpu_idx) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gpu_idx == 0 && gid == 0) {
        *dot_result_delta = 0.0;
        *dot_result_gamma = 0.0;
    }
}

}  // namespace ProfilingDiscretePipelined

int ProfilingDiscretePipelined::init(int argc, char *argv[]) {
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

    real *um_x;
    real *um_r;
    real *um_p;
    real *um_s;
    real *um_z;
    real *um_w;
    real *um_q;
    real *um_ax0;

    double *um_tmp_dot_delta1;
    double *um_tmp_dot_gamma1;

    real real_positive_one = 1.0;
    real real_negative_one = -1.0;

    CUDA_RT_CALL(cudaDeviceSynchronize());

    if (generate_random_tridiag_matrix) {
        num_rows = 10485760 * 2;
        num_cols = num_rows;

        nnz = (num_rows - 2) * 3 + 4;

        CUDA_RT_CALL(cudaMallocManaged((void **)&um_I, sizeof(int) * (num_rows + 1)));
        CUDA_RT_CALL(cudaMallocManaged((void **)&um_J, sizeof(int) * nnz));
        CUDA_RT_CALL(cudaMallocManaged((void **)&um_val, sizeof(real) * nnz));

        /* Generate a random tridiagonal symmetric matrix in CSR format */
        genTridiag(um_I, um_J, um_val, num_rows, nnz);
    } else {
        if (loadMMSparseMatrix<real>(matrix_path_char, 'd', true, &num_rows, &num_cols, &nnz,
                                     &host_val, &host_I, &host_J, true)) {
            exit(EXIT_FAILURE);
        }

        CUDA_RT_CALL(cudaMallocManaged((void **)&um_I, sizeof(int) * (num_rows + 1)));
        CUDA_RT_CALL(cudaMallocManaged((void **)&um_J, sizeof(int) * nnz));
        CUDA_RT_CALL(cudaMallocManaged((void **)&um_val, sizeof(real) * nnz));

        memcpy(um_I, host_I, sizeof(int) * (num_rows + 1));
        memcpy(um_J, host_J, sizeof(int) * nnz);
        memcpy(um_val, host_val, sizeof(real) * nnz);
    }

    if (compare_to_single_gpu) {
        CUDA_RT_CALL(cudaMallocHost(&x_ref_single_gpu, num_rows * sizeof(real)));

        single_gpu_runtime = SingleGPUDiscreteStandard::run_single_gpu(
            iter_max, um_I, um_J, um_val, x_ref_single_gpu, num_rows, nnz);

        // single_gpu_runtime = SingleGPUDiscretePipelined::run_single_gpu(
        //     iter_max, um_I, um_J, um_val, x_ref_single_gpu, num_rows, nnz);
    }

    CUDA_RT_CALL(cudaMallocManaged((void **)&um_x, sizeof(real) * num_rows));

    CUDA_RT_CALL(cudaMallocManaged((void **)&um_tmp_dot_delta1, sizeof(double)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_tmp_dot_gamma1, sizeof(double)));

    // temp memory for ConjugateGradient
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_r, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_p, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_s, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_z, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_w, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_q, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_ax0, num_rows * sizeof(real)));

    CUDA_RT_CALL(cudaMemset(um_tmp_dot_delta1, 0, sizeof(double)));
    CUDA_RT_CALL(cudaMemset(um_tmp_dot_gamma1, 0, sizeof(double)));

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

        CPU::cpuConjugateGrad(iter_max, um_I, um_J, um_val, x_ref_cpu, s_cpu, p_cpu, r_cpu, nnz,
                              num_rows, tol);
    }

    CUDA_RT_CALL(cudaDeviceSynchronize());

    cudaEvent_t atomic_add_done[num_devices];
    cudaEvent_t iteration_done[num_devices];

#pragma omp parallel num_threads(num_devices)                                                     \
    firstprivate(um_I, um_J, um_val, um_x, um_r, um_p, um_s, um_z, um_w, um_q, um_tmp_dot_delta1, \
                 um_tmp_dot_gamma1)
    {
        int gpu_idx = omp_get_thread_num();

        CUDA_RT_CALL(cudaSetDevice(gpu_idx));
        CUDA_RT_CALL(cudaFree(0));

        cudaStream_t mainStream;

        real tmp_dot_delta0;
        real tmp_dot_gamma0;

        real alpha;
        real negative_alpha;
        real beta;

        CUDA_RT_CALL(cudaEventCreateWithFlags(atomic_add_done + gpu_idx, cudaEventDisableTiming));
        CUDA_RT_CALL(cudaEventCreateWithFlags(iteration_done + gpu_idx, cudaEventDisableTiming));

        for (int gpu_idx_j = 0; gpu_idx_j < num_devices; gpu_idx_j++) {
            if (gpu_idx != gpu_idx_j) {
                CUDA_RT_CALL(cudaDeviceEnablePeerAccess(gpu_idx_j, 0));
            }
        }

#pragma omp barrier

        int sMemSize = 2 * (sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1));
        int numBlocks = (num_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

#pragma omp barrier

        CUDA_RT_CALL(cudaStreamCreate(&mainStream));

        CUDA_RT_CALL(cudaDeviceSynchronize());

#pragma omp barrier

        double start = omp_get_wtime();

        MultiGPU::initVectors<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
            um_r, um_x, num_rows, gpu_idx, num_devices);

        // ax0 = Ax0
        MultiGPU::gpuSpMV<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
            um_I, um_J, um_val, nnz, num_rows, real_positive_one, um_x, um_ax0, gpu_idx,
            num_devices);

        // r0 = b0 - ax0
        // NOTE: b is a unit vector.
        MultiGPU::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
            um_ax0, um_r, real_negative_one, num_rows, gpu_idx, num_devices);

        // w0 = Ar0
        MultiGPU::gpuSpMV<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
            um_I, um_J, um_val, nnz, num_rows, real_positive_one, um_r, um_w, gpu_idx, num_devices);

        CUDA_RT_CALL(cudaEventRecord(iteration_done[gpu_idx], mainStream));

        for (int neighbor_gpu_idx = 0; neighbor_gpu_idx < num_devices; neighbor_gpu_idx++) {
            CUDA_RT_CALL(cudaEventSynchronize(iteration_done[neighbor_gpu_idx]))
        }

        int k = 1;

        while (k <= iter_max) {
            PUSH_RANGE("Merged Dots (+Reset)", 0);

            // Two dot products => <r, r> and <r, w>
            resetLocalDotProducts<<<1, 1, 0, mainStream>>>(um_tmp_dot_delta1, um_tmp_dot_gamma1,
                                                           gpu_idx);

            // Dot
            gpuDotProductsMerged<<<numBlocks, THREADS_PER_BLOCK, sMemSize, mainStream>>>(
                um_r, um_r, um_r, um_w, num_rows, gpu_idx, num_devices, sMemSize);

            cudaStreamSynchronize(mainStream);

            POP_RANGE

            PUSH_RANGE("SpMV", 1);

            // SpMV
            MultiGPU::gpuSpMV<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
                um_I, um_J, um_val, nnz, num_rows, real_positive_one, um_w, um_q, gpu_idx,
                num_devices);

            cudaStreamSynchronize(mainStream);

            POP_RANGE

            PUSH_RANGE("Atomic Adds", 2);

            addLocalDotContributions<<<1, 1, 0, mainStream>>>(um_tmp_dot_delta1, um_tmp_dot_gamma1);

            cudaStreamSynchronize(mainStream);

            POP_RANGE

            PUSH_RANGE("Peer Sync 1", 3);

            CUDA_RT_CALL(cudaEventRecord(atomic_add_done[gpu_idx], mainStream));

            for (int neighbor_gpu_idx = 0; neighbor_gpu_idx < num_devices; neighbor_gpu_idx++) {
                CUDA_RT_CALL(cudaEventSynchronize(atomic_add_done[neighbor_gpu_idx]))
            }

            POP_RANGE

            real real_tmp_dot_delta1 = (real)*um_tmp_dot_delta1;
            real real_tmp_dot_gamma1 = (real)*um_tmp_dot_gamma1;

            if (k > 1) {
                beta = real_tmp_dot_delta1 / tmp_dot_delta0;
                alpha = real_tmp_dot_delta1 /
                        (real_tmp_dot_gamma1 - (beta / alpha) * real_tmp_dot_delta1);
            } else {
                beta = 0.0;
                alpha = real_tmp_dot_delta1 / real_tmp_dot_gamma1;
            }

            CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

            PUSH_RANGE("Saxpy 1", 4);

            // z_k = q_k + beta_k * z_(k-1)
            MultiGPU::gpuScaleVectorAndSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
                um_q, um_z, real_positive_one, beta, num_rows, gpu_idx, num_devices);

            CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

            POP_RANGE

            PUSH_RANGE("Saxpy 2", 5);

            // s_k = w_k + beta_k * s_(k-1)
            MultiGPU::gpuScaleVectorAndSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
                um_w, um_s, real_positive_one, beta, num_rows, gpu_idx, num_devices);

            CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

            POP_RANGE

            PUSH_RANGE("Saxpy 3", 6);

            // p_k = r_k = beta_k * p_(k-1)
            MultiGPU::gpuScaleVectorAndSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
                um_r, um_p, real_positive_one, beta, num_rows, gpu_idx, num_devices);

            CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

            POP_RANGE

            PUSH_RANGE("Saxpy 4", 7);

            // x_(k+1) = x_k + alpha_k * p_k
            MultiGPU::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
                um_p, um_x, alpha, num_rows, gpu_idx, num_devices);

            CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

            POP_RANGE

            negative_alpha = -alpha;

            PUSH_RANGE("Saxpy 5", 8);

            // r_(k+1) = r_k - alpha_k * s_k
            MultiGPU::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
                um_s, um_r, negative_alpha, num_rows, gpu_idx, num_devices);

            CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

            POP_RANGE

            PUSH_RANGE("Saxpy 6", 9);

            // w_(k+1) = w_k - alpha_k * z_k
            MultiGPU::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
                um_z, um_w, negative_alpha, num_rows, gpu_idx, num_devices);

            CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

            POP_RANGE

            tmp_dot_delta0 = (real)*um_tmp_dot_delta1;
            tmp_dot_gamma0 = (real)*um_tmp_dot_gamma1;

            PUSH_RANGE("Peer Sync 2", 10);

            CUDA_RT_CALL(cudaEventRecord(iteration_done[gpu_idx], 0));

            for (int neighbor_gpu_idx = 0; neighbor_gpu_idx < num_devices; neighbor_gpu_idx++) {
                CUDA_RT_CALL(cudaEventSynchronize(iteration_done[neighbor_gpu_idx]))
            }

            POP_RANGE

#pragma omp barrier

            k++;
        }

#pragma omp barrier

        double stop = omp_get_wtime();

#pragma omp master
        {
            report_results(num_rows, x_ref_single_gpu, x_ref_cpu, um_x, num_devices,
                           single_gpu_runtime, start, stop, compare_to_single_gpu, compare_to_cpu);
        }

#pragma omp barrier

        CUDA_RT_CALL(cudaStreamDestroy(mainStream));
    }

    CUDA_RT_CALL(cudaFree(um_I));
    CUDA_RT_CALL(cudaFree(um_J));
    CUDA_RT_CALL(cudaFree(um_val));
    CUDA_RT_CALL(cudaFree(um_x));
    CUDA_RT_CALL(cudaFree(um_r));
    CUDA_RT_CALL(cudaFree(um_p));
    CUDA_RT_CALL(cudaFree(um_s));
    CUDA_RT_CALL(cudaFree(um_tmp_dot_delta1));
    CUDA_RT_CALL(cudaFree(um_tmp_dot_gamma1));
    free(host_val);

    return 0;
}
