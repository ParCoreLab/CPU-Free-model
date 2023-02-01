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

#include "../../include_nvshmem/common.h"
#include "../../include_nvshmem/single-stream/standard-saxpy-overlap-nvshmem.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

namespace cg = cooperative_groups;

namespace SingleStreamStandardSaxpyOverlapNVSHMEM {

__device__ void gpuSpMV(int *rowInd, int *colInd, real *val, real alpha, real *inputVecX,
                        real *outputVecY, int row_start_idx, int chunk_size, int num_rows,
                        bool matrix_is_zero_indexed, const cg::grid_group &grid) {
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
                // If matrix is 1-indexed, need to move indices back by 1
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

__device__ void gpuDotProduct(real *vecA, real *vecB, double *local_dot_result,
                              const cg::thread_block &cta, int chunk_size,
                              const cg::grid_group &grid) {
    extern __shared__ double tmp[];

    double temp_sum = 0.0;

    for (size_t i = grid.thread_rank(); i < chunk_size; i += grid.size()) {
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

__device__ void gpuCopyVector(real *srcA, real *destB, int chunk_size, const cg::grid_group &grid) {
    for (int i = grid.thread_rank(); i < chunk_size; i += grid.size()) {
        destB[i] = srcA[i];
    }
}

__device__ void gpuSaxpy(real *x, real *y, real a, int chunk_size, bool is_comm_comp_overlap_on,
                         const cg::grid_group &grid) {
    int grid_size = grid.size() - blockDim.x * int(is_comm_comp_overlap_on);

    for (int i = grid.thread_rank(); i < chunk_size; i += grid_size) {
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
                              real *device_csrVal, real *x, real *r, real *p, real *s, real *ax0,
                              double *dot_delta1, double *dot_gamma1, int nnz, int num_rows,
                              int row_start_idx, int chunk_size, bool matrix_is_zero_indexed,
                              real tol, const int iter_max, const int sMemSize) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    int last_thread_idx = grid.size() - 1;

    real real_positive_one = 1.0;
    real real_negative_one = -1.0;

    real tmp_dot_gamma0 = 0.0;

    real real_tmp_dot_gamma1;

    real beta;
    real alpha;
    real negative_alpha;

    initVectors(r, x, row_start_idx, chunk_size, num_rows, grid);

    if (grid.thread_rank() == last_thread_idx) {
        nvshmem_barrier_all();
    }

    cg::sync(grid);

    // ax0 = Ax0
    gpuSpMV(device_csrRowIndices, device_csrColIndices, device_csrVal, real_positive_one, x, ax0,
            row_start_idx, chunk_size, num_rows, matrix_is_zero_indexed, grid);

    if (grid.thread_rank() == last_thread_idx) {
        nvshmem_barrier_all();
    }

    cg::sync(grid);

    // r0 = b0 - ax0
    // NOTE: b is a unit vector.
    // NOTE: We use r here because r is also a unit vector at first (r = b0)
    gpuSaxpy(ax0, r, real_negative_one, chunk_size, false, grid);

    cg::sync(grid);

    // p0 = r0
    gpuCopyVector(r, p, chunk_size, grid);

    cg::sync(grid);

    // First dot - gamma = r * r
    if (grid.thread_rank() == last_thread_idx) {
        *dot_gamma1 = 0.0;
    }

    cg::sync(grid);

    gpuDotProduct(r, r, dot_gamma1, cta, chunk_size, grid);

    cg::sync(grid);

    if (grid.thread_rank() == last_thread_idx) {
        nvshmem_double_sum_reduce(NVSHMEM_TEAM_WORLD, dot_gamma1, dot_gamma1, 1);
        nvshmem_barrier_all();
    }

    cg::sync(grid);

    tmp_dot_gamma0 = (real)*dot_gamma1;

    // s0 = Ar0
    gpuSpMV(device_csrRowIndices, device_csrColIndices, device_csrVal, real_positive_one, r, s,
            row_start_idx, chunk_size, num_rows, matrix_is_zero_indexed, grid);

    if (grid.thread_rank() == last_thread_idx) {
        nvshmem_barrier_all();
    }

    cg::sync(grid);

    // Second dot - delta = p * s
    if (grid.thread_rank() == last_thread_idx) {
        *dot_delta1 = 0.0;
    }

    cg::sync(grid);

    gpuDotProduct(p, s, dot_delta1, cta, chunk_size, grid);

    cg::sync(grid);

    if (grid.thread_rank() == last_thread_idx) {
        nvshmem_double_sum_reduce(NVSHMEM_TEAM_WORLD, dot_delta1, dot_delta1, 1);
        nvshmem_barrier_all();
    }

    cg::sync(grid);

    alpha = tmp_dot_gamma0 / (real)*dot_delta1;

    // NOTE!: Here k = 1 because iteration 0 is already computed
    // NOTE!: k = 0 in other versions
    int k = 1;

    while (k < iter_max) {
        negative_alpha = -alpha;

        // r_k = r_(k-1) - alpha_(k-1) * s_(k-1)
        gpuSaxpy(s, r, negative_alpha, chunk_size, false, grid);

        if (grid.thread_rank() == last_thread_idx) {
            *dot_gamma1 = 0.0;
        }

        cg::sync(grid);

        gpuDotProduct(r, r, dot_gamma1, cta, chunk_size, grid);

        cg::sync(grid);

        if (cta.group_index().x == (grid.num_blocks() - 1)) {
            nvshmemx_double_sum_reduce_block(NVSHMEM_TEAM_WORLD, dot_gamma1, dot_gamma1, 1);
        } else {
            // x_k = x_(k-1) + alpha_(k-1) * p_(k-1)
            gpuSaxpy(p, x, alpha, chunk_size, true, grid);
        }

        if (grid.thread_rank() == last_thread_idx) {
            nvshmem_barrier_all();
        }

        cg::sync(grid);

        real_tmp_dot_gamma1 = (real)*dot_gamma1;

        beta = real_tmp_dot_gamma1 / tmp_dot_gamma0;

        gpuScaleVectorAndSaxpy(r, p, real_positive_one, beta, chunk_size, grid);

        if (grid.thread_rank() == last_thread_idx) {
            nvshmem_barrier_all();
        }

        cg::sync(grid);

        gpuSpMV(device_csrRowIndices, device_csrColIndices, device_csrVal, real_positive_one, p, s,
                row_start_idx, chunk_size, num_rows, matrix_is_zero_indexed, grid);

        if (grid.thread_rank() == last_thread_idx) {
            nvshmem_barrier_all();
        }

        cg::sync(grid);

        if (grid.thread_rank() == last_thread_idx) {
            *dot_delta1 = 0.0;
        }

        cg::sync(grid);

        gpuDotProduct(p, s, dot_delta1, cta, chunk_size, grid);

        cg::sync(grid);

        if (grid.thread_rank() == last_thread_idx) {
            nvshmem_double_sum_reduce(NVSHMEM_TEAM_WORLD, dot_delta1, dot_delta1, 1);
            nvshmem_barrier_all();
        }

        cg::sync(grid);

        alpha = real_tmp_dot_gamma1 / (real)*dot_delta1;

        if (grid.thread_rank() == last_thread_idx) {
            nvshmem_barrier_all();
        }

        cg::sync(grid);

        k++;
    }
}
}  // namespace SingleStreamStandardSaxpyOverlapNVSHMEM

int SingleStreamStandardSaxpyOverlapNVSHMEM::init(
    int *device_csrRowIndices, int *device_csrColIndices, real *device_csrVal, const int num_rows,
    const int nnz, bool matrix_is_zero_indexed, const int num_devices, const int iter_max,
    real *x_final_result, const double single_gpu_runtime, bool compare_to_single_gpu,
    bool compare_to_cpu, real *x_ref_single_gpu, real *x_ref_cpu) {
    real *device_x;
    real *device_r;
    real *device_p;
    real *device_s;
    real *device_ax0;

    double *device_dot_delta1;
    double *device_dot_gamma1;

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

    // WARNING!!!
    // This was causing issues for me
    // Get rid of THREADS_PER_BLOCK
    // Use per version threadsPerBlock variable
    int threadsPerBlock = 1024;
    int sMemSize = sizeof(double) * ((threadsPerBlock / 32) + 1);

    void *kernelArgs[] = {
        (void *)&device_csrRowIndices,
        (void *)&device_csrColIndices,
        (void *)&device_csrVal,
        (void *)&device_x,
        (void *)&device_r,
        (void *)&device_p,
        (void *)&device_s,
        (void *)&device_ax0,
        (void *)&device_dot_delta1,
        (void *)&device_dot_gamma1,
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
    nvshmem_free(device_ax0);

    nvshmem_free(device_dot_delta1);
    nvshmem_free(device_dot_gamma1);

    CUDA_RT_CALL(cudaStreamDestroy(mainStream));

    return 0;
}
