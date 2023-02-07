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

#include "../../include/common.h"
#include "../../include/single-stream/pipelined-gather.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

namespace cg = cooperative_groups;

namespace SingleStreamPipelinedGather {

// Not ideal but did this to do the two SpMVs in iteration 0
// Don't want to gather the vectors for those
__device__ void gpuSpMVRemote(int *rowInd, int *colInd, real *val, real alpha, real *inputVecX,
                              real *outputVecY, int row_start_idx, int chunk_size, int num_rows,
                              bool matrix_is_zero_indexed, const cg::grid_group &grid) {
    // If we are overlapping communication with compute => 1 thread block spared for communication
    // If not => Use full grid for computation

    int mype = nvshmem_my_pe();

    for (int local_row_idx = grid.thread_rank(); local_row_idx < chunk_size;
         local_row_idx += grid.size()) {
        int global_row_idx = row_start_idx + local_row_idx;

        if (global_row_idx < num_rows) {
            int row_elem = rowInd[global_row_idx] - int(!matrix_is_zero_indexed);
            int next_row_elem = rowInd[global_row_idx + 1] - int(!matrix_is_zero_indexed);
            int num_elems_this_row = next_row_elem - row_elem;

            real output = 0.0;

            for (int j = 0; j < num_elems_this_row; j++) {
                int input_vec_elem_idx = colInd[row_elem + j] - int(!matrix_is_zero_indexed);
                int remote_pe = input_vec_elem_idx / chunk_size;

                int remote_pe_idx_offset = input_vec_elem_idx - remote_pe * chunk_size;

                // NVSHMEM calls require explicitly specifying the type
                // For now this will only work with double

                real elem_val = nvshmem_double_g(inputVecX + remote_pe_idx_offset, remote_pe);

                output += alpha * val[row_elem + j] * elem_val;
            }

            outputVecY[local_row_idx] = output;
        }
    }
}

// This will do SpMV essentially Single GPU Spmv
// inputVecX is the Gathered w vector
__device__ void gpuSpMVLocal(int *rowInd, int *colInd, real *val, real alpha, real *inputVecX,
                             real *outputVecY, int row_start_idx, int chunk_size, int num_rows,
                             bool matrix_is_zero_indexed, const cg::grid_group &grid) {
    // If we are overlapping communication with compute => 1 thread block spared for communication
    // If not => Use full grid for computation
    int grid_size = grid.size() - blockDim.x;

    for (int local_row_idx = grid.thread_rank(); local_row_idx < chunk_size;
         local_row_idx += grid_size) {
        int global_row_idx = row_start_idx + local_row_idx;

        if (global_row_idx < num_rows) {
            int row_elem = rowInd[global_row_idx] - int(!matrix_is_zero_indexed);
            int next_row_elem = rowInd[global_row_idx + 1] - int(!matrix_is_zero_indexed);
            int num_elems_this_row = next_row_elem - row_elem;

            real output = 0.0;

            for (int j = 0; j < num_elems_this_row; j++) {
                int input_vec_elem_idx = colInd[row_elem + j] - int(!matrix_is_zero_indexed);

                real elem_val = inputVecX[input_vec_elem_idx];

                output += alpha * val[row_elem + j] * elem_val;
            }

            outputVecY[local_row_idx] = output;
        }
    }
}

// Performs two dot products at the same time
// Used to perform <r, r> and <r, w> at the same time
// Can we combined the two atomicAdds somehow?
__device__ void gpuDotProductsMerged(real *vecA_delta, real *vecB_delta, real *vecA_gamma,
                                     real *vecB_gamma, double *local_dot_result_delta,
                                     double *local_dot_result_gamma, const cg::thread_block &cta,
                                     int chunk_size, const int sMemSize,
                                     const cg::grid_group &grid) {
    int grid_size = grid.size() - blockDim.x;

    // First half (up to sMemSize / 2) will be used for delta
    // Second half (from sMemSize / 2) will be used for gamma
    extern __shared__ double tmp[];

    double *tmp_delta = (double *)tmp;
    double *tmp_gamma = (double *)&tmp_delta[sMemSize / (2 * sizeof(double))];

    double temp_sum_delta = 0.0;
    double temp_sum_gamma = 0.0;

    for (int i = grid.thread_rank(); i < chunk_size; i += grid_size) {
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

__device__ void initVectors(real *r, real *x, int row_start_idx, int chunk_size, int num_rows,
                            const cg::grid_group &grid) {
    for (int local_row_idx = grid.thread_rank(); local_row_idx < chunk_size;
         local_row_idx += grid.size()) {
        int global_row_idx = row_start_idx + local_row_idx;

        if (global_row_idx < num_rows) {
            r[local_row_idx] = 1.0;
            x[local_row_idx] = 0.0;
        }
    }
}

__device__ void gpuSaxpy(real *x, real *y, real a, int chunk_size, const cg::grid_group &grid) {
    for (int i = grid.thread_rank(); i < chunk_size; i += grid.size()) {
        y[i] = a * x[i] + y[i];
    }
}

__device__ void gpuScaleVectorAndSaxpy(real *x, real *y, real a, real scale, int chunk_size,
                                       const cg::grid_group &grid) {
    for (int i = grid.thread_rank(); i < chunk_size; i += grid.size()) {
        y[i] = a * x[i] + scale * y[i];
    }
}

__global__ void __launch_bounds__(1024, 1)
    multiGpuConjugateGradient(int *device_csrRowIndices, int *device_csrColIndices,
                              real *device_csrVal, real *x, real *r, real *p, real *s, real *z,
                              real *w, real *q, real *ax0, real *gathered_w,
                              double *device_merged_dots, int nnz, int num_rows, int row_start_idx,
                              int chunk_size, bool matrix_is_zero_indexed, real tol,
                              const int iter_max, const int sMemSize) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    int last_thread_idx = grid.size() - 1;

    real real_positive_one = 1.0;
    real real_negative_one = -1.0;

    real tmp_dot_delta0 = 0.0;

    real real_tmp_dot_delta1;
    real real_tmp_dot_gamma1;

    real beta;
    real alpha;
    real negative_alpha;

    initVectors(r, x, row_start_idx, chunk_size, num_rows, grid);

    if (grid.thread_rank() == last_thread_idx) {
        nvshmem_barrier_all();
    }

    cg::sync(grid);

    // ax0 = AX0
    gpuSpMVRemote(device_csrRowIndices, device_csrColIndices, device_csrVal, real_positive_one, x,
                  ax0, row_start_idx, chunk_size, num_rows, matrix_is_zero_indexed, grid);

    if (grid.thread_rank() == last_thread_idx) {
        nvshmem_barrier_all();
    }

    cg::sync(grid);

    // r0 = b0 - ax0
    // NOTE: b is a unit vector.
    gpuSaxpy(ax0, r, real_negative_one, chunk_size, grid);

    if (grid.thread_rank() == last_thread_idx) {
        nvshmem_barrier_all();
    }

    cg::sync(grid);

    gpuSpMVRemote(device_csrRowIndices, device_csrColIndices, device_csrVal, real_positive_one, r,
                  w, row_start_idx, chunk_size, num_rows, matrix_is_zero_indexed, grid);

    if (grid.thread_rank() == last_thread_idx) {
        nvshmem_barrier_all();
    }

    cg::sync(grid);

    int k = 0;

    while (k < iter_max) {
        if (grid.thread_rank() == last_thread_idx) {
            device_merged_dots[0] = 0.0;
            device_merged_dots[1] = 0.0;
        }

        cg::sync(grid);

        // 1. Overlap a) Merged Dot with b) Gathering vector w

        if (cta.group_index().x == (grid.num_blocks() - 1)) {
            nvshmemx_double_fcollect_block(NVSHMEM_TEAM_WORLD, gathered_w, w, chunk_size);
        } else {
            gpuDotProductsMerged(r, r, r, w, &device_merged_dots[0], &device_merged_dots[1], cta,
                                 chunk_size, sMemSize, grid);
        }

        if (grid.thread_rank() == last_thread_idx) {
            nvshmem_barrier_all();
        }

        cg::sync(grid);

        // 2. Overlap a) Dot Reduction with b) SpMV Computation

        if (cta.group_index().x == (grid.num_blocks() - 1)) {
            nvshmemx_double_sum_reduce_block(NVSHMEM_TEAM_WORLD, device_merged_dots,
                                             device_merged_dots, 2);
        } else {
            gpuSpMVLocal(device_csrRowIndices, device_csrColIndices, device_csrVal,
                         real_positive_one, gathered_w, q, row_start_idx, chunk_size, num_rows,
                         matrix_is_zero_indexed, grid);
        }

        if (grid.thread_rank() == last_thread_idx) {
            nvshmem_barrier_all();
        }

        cg::sync(grid);

        real_tmp_dot_delta1 = (real)device_merged_dots[0];
        real_tmp_dot_gamma1 = (real)device_merged_dots[1];

        if (k > 1) {
            beta = real_tmp_dot_delta1 / tmp_dot_delta0;
            alpha =
                real_tmp_dot_delta1 / (real_tmp_dot_gamma1 - (beta / alpha) * real_tmp_dot_delta1);
        } else {
            beta = 0.0;
            alpha = real_tmp_dot_delta1 / real_tmp_dot_gamma1;
        }

        // z_k = q_k + beta_k * z_(k-1)
        gpuScaleVectorAndSaxpy(q, z, real_positive_one, beta, chunk_size, grid);

        // s_k = w_k + beta_k * s_(k-1)
        gpuScaleVectorAndSaxpy(w, s, real_positive_one, beta, chunk_size, grid);

        // p_k = r_k = beta_k * p_(k-1)
        gpuScaleVectorAndSaxpy(r, p, real_positive_one, beta, chunk_size, grid);

        cg::sync(grid);

        // x_(k+1) = x_k + alpha_k * p_k
        gpuSaxpy(p, x, alpha, chunk_size, grid);

        negative_alpha = -alpha;

        // r_(k+1) = r_k - alpha_k * s_k
        gpuSaxpy(s, r, negative_alpha, chunk_size, grid);

        // w_(k+1) = w_k - alpha_k * z_k
        gpuSaxpy(z, w, negative_alpha, chunk_size, grid);

        tmp_dot_delta0 = real_tmp_dot_delta1;

        if (grid.thread_rank() == last_thread_idx) {
            nvshmem_barrier_all();
        }

        cg::sync(grid);

        k++;
    }
}
}  // namespace SingleStreamPipelinedGather

int SingleStreamPipelinedGather::init(int *device_csrRowIndices, int *device_csrColIndices,
                                      real *device_csrVal, const int num_rows, const int nnz,
                                      bool matrix_is_zero_indexed, const int num_devices,
                                      const int iter_max, real *x_final_result,
                                      const double single_gpu_runtime, bool compare_to_single_gpu,
                                      bool compare_to_cpu, real *x_ref_single_gpu,
                                      real *x_ref_cpu) {
    real *device_x;
    real *device_r;
    real *device_p;
    real *device_s;
    real *device_z;
    real *device_w;
    real *device_q;
    real *device_ax0;
    real *device_gathered_w;

    double *device_merged_dots;

    int npes = nvshmem_n_pes();
    int mype = nvshmem_my_pe();

    nvshmem_barrier_all();

    cudaStream_t mainStream;
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&mainStream, cudaStreamDefault));

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

    // This will hold the gathered w vector
    // Doing this SpMV will be wholly local
    device_gathered_w = (real *)nvshmem_malloc(num_rows * sizeof(real));

    CUDA_RT_CALL(cudaMemset(device_x, 0, chunk_size * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(device_r, 0, chunk_size * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(device_p, 0, chunk_size * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(device_s, 0, chunk_size * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(device_z, 0, chunk_size * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(device_w, 0, chunk_size * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(device_q, 0, chunk_size * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(device_ax0, 0, chunk_size * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(device_gathered_w, 0, num_rows * sizeof(real)));

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

    // WARNING!!!
    // This was causing issues for me
    // Get rid of THREADS_PER_BLOCK
    // Use per version threadsPerBlock variable
    int threadsPerBlock = 1024;
    int sMemSize = 2 * sizeof(double) * ((threadsPerBlock / 32) + 1);

    void *kernelArgs[] = {
        (void *)&device_csrRowIndices,
        (void *)&device_csrColIndices,
        (void *)&device_csrVal,
        (void *)&device_x,
        (void *)&device_r,
        (void *)&device_p,
        (void *)&device_s,
        (void *)&device_z,
        (void *)&device_w,
        (void *)&device_q,
        (void *)&device_ax0,
        (void *)&device_gathered_w,
        (void *)&device_merged_dots,
        (void *)&nnz,
        (void *)&num_rows,
        (void *)&row_start_global_idx,
        (void *)&chunk_size,
        (void *)&matrix_is_zero_indexed,
        (void *)&tol,
        (void *)&iter_max,
        (void *)&sMemSize,
    };

    int numBlocks = 0;

    nvshmemx_collective_launch_query_gridsize((void *)multiGpuConjugateGradient, threadsPerBlock,
                                              kernelArgs, sMemSize, &numBlocks);

    nvshmem_barrier_all();

    double start = MPI_Wtime();

    nvshmemx_collective_launch((void *)multiGpuConjugateGradient, numBlocks, threadsPerBlock,
                               kernelArgs, sMemSize, mainStream);

    nvshmemx_barrier_all_on_stream(mainStream);
    CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

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

    return 0;
}
