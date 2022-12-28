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

#include "../../include_nvshmem/common.h"
#include "../../include_nvshmem/profiling/discrete-pipelined-nvshmem.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace ProfilingDiscretePipelinedNVSHMEM {

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

}  // namespace ProfilingDiscretePipelinedNVSHMEM

int ProfilingDiscretePipelinedNVSHMEM::init(int argc, char *argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 10000);
    std::string matrix_path_str = get_argval<std::string>(argv, argv + argc, "-matrix_path", "");
    const bool compare_to_single_gpu = get_arg(argv, argv + argc, "-compare-single-gpu");
    const bool compare_to_cpu = get_arg(argv, argv + argc, "-compare-cpu");

    char *matrix_path_char = const_cast<char *>(matrix_path_str.c_str());
    bool generate_random_tridiag_matrix = matrix_path_str.empty();

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
    real *x_final_result = NULL;

    real *s_cpu = NULL;
    real *r_cpu = NULL;
    real *p_cpu = NULL;
    real *x_ref_cpu = NULL;

    int *device_I = NULL;
    int *device_J = NULL;
    real *device_val = NULL;

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

    double *device_dot_delta1;
    double *device_dot_gamma1;
    double host_dot_gamma1;
    double host_dot_delta1;

    real real_positive_one = 1.0;
    real real_negative_one = -1.0;

    int rank = 0, size = 1;
    MPI_CALL(MPI_Init(&argc, &argv));
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));

    int local_rank = -1;
    int local_size = 1;
    {
        MPI_Comm local_comm;
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                     &local_comm));

        MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));
        MPI_CALL(MPI_Comm_size(local_comm, &local_size));

        MPI_CALL(MPI_Comm_free(&local_comm));
    }

    if (1 < num_devices && num_devices < local_size) {
        fprintf(stderr,
                "ERROR Number of visible devices (%d) is less than number of ranks on the "
                "node (%d)!\n",
                num_devices, local_size);
        MPI_CALL(MPI_Finalize());
        return 1;
    }

    if (1 == num_devices) {
        // Only 1 device visible, assuming GPU affinity is handled via CUDA_VISIBLE_DEVICES
        CUDA_RT_CALL(cudaSetDevice(0));
    } else {
        CUDA_RT_CALL(cudaSetDevice(local_rank));
    }

    CUDA_RT_CALL(cudaFree(0));

    MPI_Comm mpi_comm;
    nvshmemx_init_attr_t attr;

    mpi_comm = MPI_COMM_WORLD;
    attr.mpi_comm = &mpi_comm;

    if (generate_random_tridiag_matrix) {
        num_rows = 10485760 * 2;
        num_cols = num_rows;

        nnz = (num_rows - 2) * 3 + 4;

        host_I = (int *)malloc(sizeof(int) * (num_rows + 1));
        host_J = (int *)malloc(sizeof(int) * nnz);
        host_val = (real *)malloc(sizeof(real) * nnz);

        /* Generate a random tridiagonal symmetric matrix in CSR format */
        genTridiag(host_I, host_J, host_val, num_rows, nnz);
    } else {
        if (loadMMSparseMatrix<real>(matrix_path_char, 'd', true, &num_rows, &num_cols, &nnz,
                                     &host_val, &host_I, &host_J, true)) {
            exit(EXIT_FAILURE);
        }
    }

    CUDA_RT_CALL(cudaMalloc((void **)&device_I, sizeof(int) * (num_rows + 1)));
    CUDA_RT_CALL(cudaMalloc((void **)&device_J, sizeof(int) * nnz));
    CUDA_RT_CALL(cudaMalloc((void **)&device_val, sizeof(real) * nnz));

    CUDA_RT_CALL(
        cudaMemcpy(device_I, host_I, sizeof(int) * (num_rows + 1), cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(device_J, host_J, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(device_val, host_val, sizeof(real) * nnz, cudaMemcpyHostToDevice));

    // Set symmetric heap size for nvshmem based on problem size
    // Its default value in nvshmem is 1 GB which is not sufficient
    // for large mesh sizes
    long long unsigned int mesh_size_per_rank = num_rows / size + (num_rows % size != 0);

    long long unsigned int required_symmetric_heap_size =
        8 * mesh_size_per_rank * sizeof(real) * 1.1;

    char *value = getenv("NVSHMEM_SYMMETRIC_SIZE");
    if (value) { /* env variable is set */
        long long unsigned int size_env = parse_nvshmem_symmetric_size(value);
        if (size_env < required_symmetric_heap_size) {
            fprintf(stderr,
                    "ERROR: Minimum NVSHMEM_SYMMETRIC_SIZE = %lluB, Current "
                    "NVSHMEM_SYMMETRIC_SIZE=%s\n",
                    required_symmetric_heap_size, value);
            MPI_CALL(MPI_Finalize());
            return -1;
        }
    } else {
        char symmetric_heap_size_str[100];
        sprintf(symmetric_heap_size_str, "%llu", required_symmetric_heap_size);

        // if (rank == 0) {
        //     printf("Setting environment variable NVSHMEM_SYMMETRIC_SIZE = %llu\n",
        //            required_symmetric_heap_size);
        // }

        setenv("NVSHMEM_SYMMETRIC_SIZE", symmetric_heap_size_str, 1);
    }
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

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

    device_dot_delta1 = (double *)nvshmem_malloc(sizeof(double));
    device_dot_gamma1 = (double *)nvshmem_malloc(sizeof(double));

    CUDA_RT_CALL(cudaMemset(device_dot_delta1, 0, sizeof(double)));
    CUDA_RT_CALL(cudaMemset(device_dot_gamma1, 0, sizeof(double)));

    // Calculate local domain boundaries
    int row_start_global_idx = mype * chunk_size;      // My start index in the global array
    int row_end_global_idx = (mype + 1) * chunk_size;  // My end index in the global array

    row_end_global_idx = std::min(row_end_global_idx, num_rows);

    if (compare_to_single_gpu) {
        CUDA_RT_CALL(cudaMallocHost(&x_ref_single_gpu, num_rows * sizeof(real)));

        single_gpu_runtime = SingleGPUDiscreteStandard::run_single_gpu(
            iter_max, device_I, device_J, device_val, x_ref_single_gpu, num_rows, nnz);

        // single_gpu_runtime = SingleGPUDiscretePipelined::run_single_gpu(
        //     iter_max, device_I, device_J, device_val, x_ref_single_gpu, num_rows, nnz);
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

        CPU::cpuConjugateGrad(iter_max, host_I, host_J, host_val, x_ref_cpu, s_cpu, p_cpu, r_cpu,
                              nnz, num_rows, tol);
    }

    CUDA_RT_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    int sMemSize = 2 * sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1);
    int numBlocks = (chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    nvshmem_barrier_all();

    double start = MPI_Wtime();

    NVSHMEM::initVectors<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
        device_r, device_x, row_start_global_idx, chunk_size, num_rows);

    nvshmemx_barrier_all_on_stream(mainStream);

    // ax0 = Ax0
    NVSHMEM::gpuSpMV<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
        device_I, device_J, device_val, real_positive_one, device_x, device_ax0,
        row_start_global_idx, chunk_size, num_rows);

    nvshmemx_barrier_all_on_stream(mainStream);

    // r0 = b0 - ax0
    // NOTE: b is a unit vector.
    NVSHMEM::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
        device_ax0, device_r, real_negative_one, chunk_size);

    nvshmemx_barrier_all_on_stream(mainStream);

    // w0 = Ar0
    NVSHMEM::gpuSpMV<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
        device_I, device_J, device_val, real_positive_one, device_r, device_w, row_start_global_idx,
        chunk_size, num_rows);

    nvshmemx_barrier_all_on_stream(mainStream);

    int k = 1;

    while (k <= iter_max) {
        PUSH_RANGE("Merged Dots (+Reset)", 0);

        resetLocalDotProducts<<<1, 1, 0, mainStream>>>(device_dot_delta1, device_dot_gamma1);

        // Dot
        gpuDotProductsMerged<<<numBlocks, THREADS_PER_BLOCK, sMemSize, mainStream>>>(
            device_r, device_r, device_r, device_w, device_dot_delta1, device_dot_gamma1,
            chunk_size, sMemSize);

        CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

        POP_RANGE

        PUSH_RANGE("SpMV", 1);

        // SpMV
        NVSHMEM::gpuSpMV<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
            device_I, device_J, device_val, real_positive_one, device_w, device_q,
            row_start_global_idx, chunk_size, num_rows);

        CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

        POP_RANGE

        PUSH_RANGE("NVSHMEM Barrier 1 (After SpMV)", 2);

        nvshmemx_barrier_all_on_stream(mainStream);

        CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

        POP_RANGE

        // NOTE: Instead of doing this could have the local dots be in contiguous locations
        // And the use the same NVSHMEM call to do both at the same time

        PUSH_RANGE("Global Reductions (+Barrier)", 3);

        nvshmemx_double_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, device_dot_delta1,
                                             device_dot_delta1, 1, mainStream);

        nvshmemx_double_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, device_dot_gamma1,
                                             device_dot_gamma1, 1, mainStream);

        // Using nvshmem_barrier_all() here seems to cause a deadlock
        // Wonder why?
        // Because two reductions are enqued to same stream back to back?
        // In any case, should use one contiguous array for reductions
        nvshmemx_barrier_all_on_stream(mainStream);

        CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

        POP_RANGE

        PUSH_RANGE("Memcpy Dots To Host", 4);

        CUDA_RT_CALL(cudaMemcpyAsync(&host_dot_delta1, device_dot_delta1, sizeof(double),
                                     cudaMemcpyDeviceToHost, mainStream));

        CUDA_RT_CALL(cudaMemcpyAsync(&host_dot_gamma1, device_dot_gamma1, sizeof(double),
                                     cudaMemcpyDeviceToHost, mainStream));

        CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

        POP_RANGE

        real real_tmp_dot_delta1 = (real)host_dot_delta1;
        real real_tmp_dot_gamma1 = (real)host_dot_gamma1;

        if (k > 1) {
            beta = real_tmp_dot_delta1 / tmp_dot_delta0;
            alpha =
                real_tmp_dot_delta1 / (real_tmp_dot_gamma1 - (beta / alpha) * real_tmp_dot_delta1);
        } else {
            beta = 0.0;
            alpha = real_tmp_dot_delta1 / real_tmp_dot_gamma1;
        }

        PUSH_RANGE("Saxpy 1", 5);

        // z_k = q_k + beta_k * z_(k-1)
        NVSHMEM::gpuScaleVectorAndSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
            device_q, device_z, real_positive_one, beta, chunk_size);

        CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

        POP_RANGE

        PUSH_RANGE("Saxpy 2", 6);

        // s_k = w_k + beta_k * s_(k-1)
        NVSHMEM::gpuScaleVectorAndSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
            device_w, device_s, real_positive_one, beta, chunk_size);

        CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

        POP_RANGE

        PUSH_RANGE("Saxpy 3", 7);

        // p_k = r_k = beta_k * p_(k-1)
        NVSHMEM::gpuScaleVectorAndSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
            device_r, device_p, real_positive_one, beta, chunk_size);

        CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

        POP_RANGE

        PUSH_RANGE("Saxpy 4", 8);

        // x_(k+1) = x_k + alpha_k * p_k
        NVSHMEM::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(device_p, device_x,
                                                                           alpha, chunk_size);

        CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

        POP_RANGE

        negative_alpha = -alpha;

        PUSH_RANGE("Saxpy 5", 9);

        // r_(k+1) = r_k - alpha_k * s_k
        NVSHMEM::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
            device_s, device_r, negative_alpha, chunk_size);

        CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

        POP_RANGE

        PUSH_RANGE("Saxpy 6", 10);

        // w_(k+1) = w_k - alpha_k * z_k
        NVSHMEM::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
            device_z, device_w, negative_alpha, chunk_size);

        CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

        POP_RANGE

        tmp_dot_delta0 = real_tmp_dot_delta1;

        PUSH_RANGE("NVSHMEM Barrier 2 (End of Iteration)", 11);

        nvshmemx_barrier_all_on_stream(mainStream);
        CUDA_RT_CALL(cudaStreamSynchronize(mainStream));

        POP_RANGE

        k++;
    }

    nvshmemx_barrier_all_on_stream(mainStream);
    CUDA_RT_CALL(cudaDeviceSynchronize());

    double stop = MPI_Wtime();

    if (compare_to_single_gpu || compare_to_cpu) {
        CUDA_RT_CALL(cudaMallocHost(&x_final_result, num_rows * sizeof(real)));

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

    nvshmem_free(device_dot_delta1);
    nvshmem_free(device_dot_gamma1);

    CUDA_RT_CALL(cudaStreamDestroy(mainStream));

    CUDA_RT_CALL(cudaFree(device_I));
    CUDA_RT_CALL(cudaFree(device_J));
    CUDA_RT_CALL(cudaFree(device_val));

    free(host_I);
    free(host_J);
    free(host_val);

    if (compare_to_single_gpu || compare_to_cpu) {
        cudaFreeHost(x_final_result);

        if (compare_to_single_gpu) {
            cudaFreeHost(x_ref_single_gpu);
        }

        if (compare_to_cpu) {
            cudaFreeHost(x_ref_cpu);
        }
    }

    nvshmem_finalize();
    MPI_CALL(MPI_Finalize());

    return 0;
}