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

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "../../include_nvshmem/baseline/discrete-standard-nvshmem.cuh"
#include "../../include_nvshmem/common.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace BaselineDiscreteStandardNVSHMEM {

__global__ void gpuDotProduct(real *vecA, real *vecB, double *local_dot_result, int chunk_size) {
    cg::thread_block cta = cg::this_thread_block();

    size_t grid_rank = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * blockDim.x;

    extern __shared__ double tmp[];

    double temp_sum = 0.0;

    for (size_t i = grid_rank; i < chunk_size; i += grid_size) {
        temp_sum += (double)(vecA[i] * vecB[i]);
    }

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    temp_sum = cg::reduce(tile32, temp_sum, cg::plus<double>());

    if (tile32.thread_rank() == 0) {
        tmp[tile32.meta_group_rank()] = temp_sum;
    }

    cg::sync(cta);

    if (tile32.meta_group_rank() == 0) {
        temp_sum =
            tile32.thread_rank() < tile32.meta_group_size() ? tmp[tile32.thread_rank()] : 0.0;
        temp_sum = cg::reduce(tile32, temp_sum, cg::plus<double>());

        if (tile32.thread_rank() == 0) {
            atomicAdd(local_dot_result, temp_sum);
        }
    }
}

__global__ void resetLocalDotProduct(double *dot_result) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        *dot_result = 0.0;
    }
}

}  // namespace BaselineDiscreteStandardNVSHMEM

int BaselineDiscreteStandardNVSHMEM::init(int *device_csrRowIndices, int *device_csrColIndices,
                                          real *device_csrVal, const int num_rows, const int nnz,
                                          bool matrix_is_zero_indexed, const int num_devices,
                                          const int iter_max, real *x_final_result,
                                          const double single_gpu_runtime,
                                          bool compare_to_single_gpu, bool compare_to_cpu,
                                          real *x_ref_single_gpu, real *x_ref_cpu) {
    real *device_x;
    real *device_r;
    real *device_p;
    real *device_s;
    real *device_ax0;

    real alpha;
    real negative_alpha;
    real beta;

    real tmp_dot_gamma0;

    double *device_dot_delta1;
    double *device_dot_gamma1;
    double host_dot_gamma1;
    double host_dot_delta1;

    real real_positive_one = 1.0;
    real real_negative_one = -1.0;

    int npes = nvshmem_n_pes();
    int mype = nvshmem_my_pe();

    nvshmem_barrier_all();

    cudaStream_t mainStream;
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&mainStream, cudaStreamNonBlocking));

    nvshmem_barrier_all();

    // Load balancing this way isn't ideal
    // On kernel side, we need to calculate PE element belong to
    // Naive load balancing like this makes PE calculation on kernel side easier
    int chunk_size = num_rows / npes + (num_rows % npes != 0);

    device_x = (real *)nvshmem_malloc(chunk_size * sizeof(real));
    device_r = (real *)nvshmem_malloc(chunk_size * sizeof(real));
    device_p = (real *)nvshmem_malloc(chunk_size * sizeof(real));
    device_s = (real *)nvshmem_malloc(chunk_size * sizeof(real));
    device_ax0 = (real *)nvshmem_malloc(chunk_size * sizeof(real));

    CUDA_RT_CALL(cudaMemset(device_x, 0, chunk_size * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(device_r, 0, chunk_size * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(device_p, 0, chunk_size * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(device_s, 0, chunk_size * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(device_ax0, 0, chunk_size * sizeof(real)));

    device_dot_delta1 = (double *)nvshmem_malloc(sizeof(double));
    device_dot_gamma1 = (double *)nvshmem_malloc(sizeof(double));

    CUDA_RT_CALL(cudaMemset(device_dot_delta1, 0, sizeof(double)));
    CUDA_RT_CALL(cudaMemset(device_dot_gamma1, 0, sizeof(double)));

    // Calculate local domain boundaries
    int row_start_global_idx = mype * chunk_size;      // My start index in the global array
    int row_end_global_idx = (mype + 1) * chunk_size;  // My end index in the global array

    row_end_global_idx = std::min(row_end_global_idx, num_rows);

    CUDA_RT_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    int threadsPerBlock = 1024;
    int sMemSize = sizeof(double) * ((threadsPerBlock / 32) + 1);
    int numBlocks = (chunk_size + threadsPerBlock - 1) / threadsPerBlock;

    nvshmem_barrier_all();

    double start = MPI_Wtime();

    NVSHMEM::initVectors<<<numBlocks, threadsPerBlock, 0, mainStream>>>(
        device_r, device_x, row_start_global_idx, chunk_size, num_rows);

    nvshmemx_barrier_all_on_stream(mainStream);

    // ax0 = Ax0
    NVSHMEM::gpuSpMV<<<numBlocks, threadsPerBlock, 0, mainStream>>>(
        device_csrRowIndices, device_csrColIndices, device_csrVal, real_positive_one, device_x,
        device_ax0, row_start_global_idx, chunk_size, num_rows, matrix_is_zero_indexed);

    nvshmemx_barrier_all_on_stream(mainStream);

    // r0 = b0 - ax0
    // NOTE: b is a unit vector.
    NVSHMEM::gpuSaxpy<<<numBlocks, threadsPerBlock, 0, mainStream>>>(device_ax0, device_r,
                                                                     real_negative_one, chunk_size);

    // // p0 = r0
    NVSHMEM::gpuCopyVector<<<numBlocks, threadsPerBlock, 0, mainStream>>>(device_r, device_p,
                                                                          chunk_size);

    // Do we need to reset the local dot here?
    resetLocalDotProduct<<<1, 1, 0, mainStream>>>(device_dot_gamma1);

    gpuDotProduct<<<numBlocks, threadsPerBlock, sMemSize, mainStream>>>(
        device_r, device_r, device_dot_gamma1, chunk_size);

    // Do global reduction to add up local dot products
    nvshmemx_double_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, device_dot_gamma1, device_dot_gamma1,
                                         1, mainStream);

    nvshmemx_barrier_all_on_stream(mainStream);

    CUDA_RT_CALL(cudaMemcpyAsync(&host_dot_gamma1, device_dot_gamma1, sizeof(double),
                                 cudaMemcpyDeviceToHost, mainStream));

    CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

    tmp_dot_gamma0 = (real)host_dot_gamma1;

    int k = 0;

    while (k < iter_max) {
        // SpMV
        NVSHMEM::gpuSpMV<<<numBlocks, threadsPerBlock, 0, mainStream>>>(
            device_csrRowIndices, device_csrColIndices, device_csrVal, real_positive_one, device_p,
            device_s, row_start_global_idx, chunk_size, num_rows, matrix_is_zero_indexed);

        nvshmemx_barrier_all_on_stream(mainStream);

        resetLocalDotProduct<<<1, 1, 0, mainStream>>>(device_dot_delta1);

        gpuDotProduct<<<numBlocks, threadsPerBlock, sMemSize, mainStream>>>(
            device_p, device_s, device_dot_delta1, chunk_size);

        // Do global reduction to add up local dot products
        nvshmemx_double_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, device_dot_delta1,
                                             device_dot_delta1, 1, mainStream);

        nvshmemx_barrier_all_on_stream(mainStream);

        CUDA_RT_CALL(cudaMemcpyAsync(&host_dot_delta1, device_dot_delta1, sizeof(double),
                                     cudaMemcpyDeviceToHost, mainStream));

        CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

        alpha = tmp_dot_gamma0 / ((real)host_dot_delta1);

        // x_(k+1) = x_k + alpha_k * p_k
        NVSHMEM::gpuSaxpy<<<numBlocks, threadsPerBlock, 0, mainStream>>>(device_p, device_x, alpha,
                                                                         chunk_size);

        negative_alpha = -alpha;

        // r_(k+1) = r_k - alpha_k * s
        NVSHMEM::gpuSaxpy<<<numBlocks, threadsPerBlock, 0, mainStream>>>(
            device_s, device_r, negative_alpha, chunk_size);

        resetLocalDotProduct<<<1, 1, 0, mainStream>>>(device_dot_gamma1);

        gpuDotProduct<<<numBlocks, threadsPerBlock, sMemSize, mainStream>>>(
            device_r, device_r, device_dot_gamma1, chunk_size);

        // Do global reduction to add up local dot products
        nvshmemx_double_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, device_dot_gamma1,
                                             device_dot_gamma1, 1, mainStream);

        nvshmemx_barrier_all_on_stream(mainStream);

        CUDA_RT_CALL(cudaMemcpyAsync(&host_dot_gamma1, device_dot_gamma1, sizeof(double),
                                     cudaMemcpyDeviceToHost, mainStream));

        CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

        beta = ((real)host_dot_gamma1) / tmp_dot_gamma0;

        // p_(k+1) = r_(k+1) = beta_k * p_(k)
        NVSHMEM::gpuScaleVectorAndSaxpy<<<numBlocks, threadsPerBlock, 0, mainStream>>>(
            device_r, device_p, real_positive_one, beta, chunk_size);

        tmp_dot_gamma0 = (real)host_dot_gamma1;

        nvshmemx_barrier_all_on_stream(mainStream);
        CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

        k++;
    }

    nvshmemx_barrier_all_on_stream(mainStream);
    CUDA_RT_CALL(cudaDeviceSynchronize());

    double stop = MPI_Wtime();

    if (compare_to_single_gpu || compare_to_cpu) {
        // Need to do this when when num_rows % npes != 0
        int num_elems_to_copy = row_end_global_idx - row_start_global_idx;

        CUDA_RT_CALL(cudaMemcpy(x_final_result + row_start_global_idx, device_x,
                                num_elems_to_copy * sizeof(real), cudaMemcpyDeviceToHost));
    }

    bool result_correct_single_gpu = true;
    bool result_correct_cpu = true;

    report_errors(num_rows, x_ref_single_gpu, x_ref_cpu, x_final_result, row_start_global_idx,
                  row_end_global_idx, npes, single_gpu_runtime, start, stop, compare_to_single_gpu,
                  compare_to_cpu, result_correct_single_gpu, result_correct_cpu);

    nvshmem_barrier_all();

    if (mype == 0) {
        report_runtime(npes, single_gpu_runtime, start, stop, result_correct_single_gpu,
                       result_correct_cpu, compare_to_single_gpu);
    }

    nvshmem_free(device_x);
    nvshmem_free(device_r);
    nvshmem_free(device_p);
    nvshmem_free(device_s);
    nvshmem_free(device_ax0);

    nvshmem_free(device_dot_delta1);
    nvshmem_free(device_dot_gamma1);

    CUDA_RT_CALL(cudaStreamDestroy(mainStream));

    return 0;
}