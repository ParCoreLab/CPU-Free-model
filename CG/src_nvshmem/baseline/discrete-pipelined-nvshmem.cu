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
#include <iostream>
#include <map>
#include <set>
#include <utility>

#include <omp.h>

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "../../include_nvshmem/baseline/discrete-pipelined-nvshmem.cuh"
#include "../../include_nvshmem/common.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace BaselineDiscretePipelinedNVSHMEM {

__global__ void gpuDotProductsMerged(real *vecA_delta, real *vecB_delta, real *vecA_gamma,
                                     real *vecB_gamma, double *local_dot_result_delta,
                                     double *local_dot_result_gamma, int chunk_size,
                                     const int sMemSize) {
    cg::thread_block cta = cg::this_thread_block();

    size_t grid_rank = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * blockDim.x;

    extern __shared__ double tmp[];

    double *tmp_delta = (double *)tmp;
    double *tmp_gamma = (double *)&tmp_delta[sMemSize / (2 * sizeof(double))];

    double temp_sum_delta = 0.0;
    double temp_sum_gamma = 0.0;

    for (size_t i = grid_rank; i < chunk_size; i += grid_size) {
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
            atomicAdd(local_dot_result_delta, temp_sum_delta);
            atomicAdd(local_dot_result_gamma, temp_sum_gamma);
        }
    }
}

__global__ void resetLocalDotProducts(double *dot_result_delta, double *dot_result_gamma) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        *dot_result_delta = 0.0;
        *dot_result_gamma = 0.0;
    }
}

}  // namespace BaselineDiscretePipelinedNVSHMEM

int BaselineDiscretePipelinedNVSHMEM::init(int *device_csrRowIndices, int *device_csrColIndices,
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
    real *device_z;
    real *device_w;
    real *device_q;
    real *device_ax0;

    real alpha;
    real negative_alpha;
    real beta;

    real tmp_dot_delta0;

    double *device_merged_dots;

    double *host_merged_dots = (double *)malloc(2 * sizeof(double));

    real real_positive_one = 1.0;
    real real_negative_one = -1.0;

    int npes = nvshmem_n_pes();
    int mype = nvshmem_my_pe();

    nvshmem_barrier_all();

    cudaStream_t mainStream;
    cudaStream_t SpMVStream;
    cudaStream_t communicationStream;

    CUDA_RT_CALL(cudaStreamCreateWithFlags(&mainStream, cudaStreamNonBlocking));
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&SpMVStream, cudaStreamNonBlocking));
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&communicationStream, cudaStreamNonBlocking));

    nvshmem_barrier_all();

    // Load balancing this way isn't ideal
    // On kernel side, we need to calculate PE element belong to
    // Naive load balancing like this makes PE calculation on kernel side easier
    int chunk_size = num_rows / npes + (num_rows % npes != 0);

    device_x = (real *)nvshmem_malloc(chunk_size * sizeof(real));
    device_r = (real *)nvshmem_malloc(chunk_size * sizeof(real));
    device_p = (real *)nvshmem_malloc(chunk_size * sizeof(real));
    device_s = (real *)nvshmem_malloc(chunk_size * sizeof(real));
    device_z = (real *)nvshmem_malloc(chunk_size * sizeof(real));
    device_w = (real *)nvshmem_malloc(chunk_size * sizeof(real));
    device_q = (real *)nvshmem_malloc(chunk_size * sizeof(real));
    device_ax0 = (real *)nvshmem_malloc(chunk_size * sizeof(real));

    CUDA_RT_CALL(cudaMemset(device_x, 0, chunk_size * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(device_r, 0, chunk_size * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(device_p, 0, chunk_size * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(device_s, 0, chunk_size * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(device_z, 0, chunk_size * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(device_w, 0, chunk_size * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(device_q, 0, chunk_size * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(device_ax0, 0, chunk_size * sizeof(real)));

    // device_merged_dots[0] is dot delta
    // device_merged_dots[1] is dot gamma
    device_merged_dots = (double *)nvshmem_malloc(2 * sizeof(double));

    CUDA_RT_CALL(cudaMemset(device_merged_dots, 0, 2 * sizeof(double)));

    // Calculate local domain boundaries
    int row_start_global_idx = mype * chunk_size;      // My start index in the global array
    int row_end_global_idx = (mype + 1) * chunk_size;  // My end index in the global array

    row_end_global_idx = std::min(row_end_global_idx, num_rows);

    CUDA_RT_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    int threadsPerBlock = 1024;
    int sMemSize = 2 * sizeof(double) * ((threadsPerBlock / 32) + 1);
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

    nvshmemx_barrier_all_on_stream(mainStream);

    // w0 = Ar0
    NVSHMEM::gpuSpMV<<<numBlocks, threadsPerBlock, 0, mainStream>>>(
        device_csrRowIndices, device_csrColIndices, device_csrVal, real_positive_one, device_r,
        device_w, row_start_global_idx, chunk_size, num_rows, matrix_is_zero_indexed);

    nvshmemx_barrier_all_on_stream(mainStream);

    int k = 0;

    while (k < iter_max) {
        resetLocalDotProducts<<<1, 1, 0, mainStream>>>(&device_merged_dots[0],
                                                       &device_merged_dots[1]);

        // Dot
        gpuDotProductsMerged<<<numBlocks, threadsPerBlock, sMemSize, mainStream>>>(
            device_r, device_r, device_r, device_w, &device_merged_dots[0], &device_merged_dots[1],
            chunk_size, sMemSize);

        CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

        // SpMV
        NVSHMEM::gpuSpMV<<<numBlocks, threadsPerBlock, 0, SpMVStream>>>(
            device_csrRowIndices, device_csrColIndices, device_csrVal, real_positive_one, device_w,
            device_q, row_start_global_idx, chunk_size, num_rows, matrix_is_zero_indexed);

        // NOTE: Instead of doing this could have the local dots be in contiguous locations
        // And the use the same NVSHMEM call to do both at the same time

        nvshmemx_double_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, device_merged_dots,
                                             device_merged_dots, 2, communicationStream);

        // Using nvshmem_barrier_all() here seems to cause a deadlock
        // Wonder why?
        // Because two reductions are enqued to same stream back to back?
        // In any case, should use one contiguous array for reductions
        nvshmemx_barrier_all_on_stream(communicationStream);

        CUDA_RT_CALL(cudaMemcpyAsync(host_merged_dots, device_merged_dots, 2 * sizeof(double),
                                     cudaMemcpyDeviceToHost, communicationStream));

        CUDA_RT_CALL(cudaStreamSynchronize(communicationStream));

        real real_tmp_dot_delta1 = (real)host_merged_dots[0];
        real real_tmp_dot_gamma1 = (real)host_merged_dots[1];

        if (k > 1) {
            beta = real_tmp_dot_delta1 / tmp_dot_delta0;
            alpha =
                real_tmp_dot_delta1 / (real_tmp_dot_gamma1 - (beta / alpha) * real_tmp_dot_delta1);
        } else {
            beta = 0.0;
            alpha = real_tmp_dot_delta1 / real_tmp_dot_gamma1;
        }

        nvshmemx_barrier_all_on_stream(SpMVStream);
        CUDA_RT_CALL(cudaStreamSynchronize(SpMVStream));

        // z_k = q_k + beta_k * z_(k-1)
        NVSHMEM::gpuScaleVectorAndSaxpy<<<numBlocks, threadsPerBlock, 0, mainStream>>>(
            device_q, device_z, real_positive_one, beta, chunk_size);

        // s_k = w_k + beta_k * s_(k-1)
        NVSHMEM::gpuScaleVectorAndSaxpy<<<numBlocks, threadsPerBlock, 0, mainStream>>>(
            device_w, device_s, real_positive_one, beta, chunk_size);

        // p_k = r_k = beta_k * p_(k-1)
        NVSHMEM::gpuScaleVectorAndSaxpy<<<numBlocks, threadsPerBlock, 0, mainStream>>>(
            device_r, device_p, real_positive_one, beta, chunk_size);

        // x_(k+1) = x_k + alpha_k * p_k
        NVSHMEM::gpuSaxpy<<<numBlocks, threadsPerBlock, 0, mainStream>>>(device_p, device_x, alpha,
                                                                         chunk_size);

        negative_alpha = -alpha;

        // r_(k+1) = r_k - alpha_k * s_k
        NVSHMEM::gpuSaxpy<<<numBlocks, threadsPerBlock, 0, mainStream>>>(
            device_s, device_r, negative_alpha, chunk_size);

        // w_(k+1) = w_k - alpha_k * z_k
        NVSHMEM::gpuSaxpy<<<numBlocks, threadsPerBlock, 0, mainStream>>>(
            device_z, device_w, negative_alpha, chunk_size);

        tmp_dot_delta0 = real_tmp_dot_delta1;

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
    nvshmem_free(device_z);
    nvshmem_free(device_w);
    nvshmem_free(device_q);
    nvshmem_free(device_ax0);

    nvshmem_free(device_merged_dots);

    CUDA_RT_CALL(cudaStreamDestroy(mainStream));
    CUDA_RT_CALL(cudaStreamDestroy(SpMVStream));
    CUDA_RT_CALL(cudaStreamDestroy(communicationStream));

    free(host_merged_dots);

    return 0;
}