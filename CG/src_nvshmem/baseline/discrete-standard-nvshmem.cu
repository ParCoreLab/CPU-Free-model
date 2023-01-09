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

int BaselineDiscreteStandardNVSHMEM::init(int argc, char *argv[]) {
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
    bool matrix_is_zero_indexed;

    int *host_csrRowIndices = NULL;
    int *host_csrColIndices = NULL;
    real *host_csrVal = NULL;

    real *x_ref_single_gpu = NULL;
    real *x_final_result = NULL;

    real *s_cpu = NULL;
    real *r_cpu = NULL;
    real *p_cpu = NULL;
    real *x_ref_cpu = NULL;

    int *device_csrRowIndices = NULL;
    int *device_csrColIndices = NULL;
    real *device_csrVal = NULL;

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

        host_csrRowIndices = (int *)malloc(sizeof(int) * (num_rows + 1));
        host_csrColIndices = (int *)malloc(sizeof(int) * nnz);
        host_csrVal = (real *)malloc(sizeof(real) * nnz);

        /* Generate a random tridiagonal symmetric matrix in CSR format */
        genTridiag(host_csrRowIndices, host_csrColIndices, host_csrVal, num_rows, nnz);
    } else {
        if (loadMMSparseMatrix<real>(matrix_path_char, 'd', true, &num_rows, &num_cols, &nnz,
                                     &host_csrVal, &host_csrRowIndices, &host_csrColIndices,
                                     true)) {
            exit(EXIT_FAILURE);
        }
    }

    // Check if matrix is 0 or 1 indexed
    int index_base = host_csrRowIndices[0];

    if (index_base == 1) {
        matrix_is_zero_indexed = false;
    } else if (index_base == 0) {
        matrix_is_zero_indexed = true;
    }

    CUDA_RT_CALL(cudaMalloc((void **)&device_csrRowIndices, sizeof(int) * (num_rows + 1)));
    CUDA_RT_CALL(cudaMalloc((void **)&device_csrColIndices, sizeof(int) * nnz));
    CUDA_RT_CALL(cudaMalloc((void **)&device_csrVal, sizeof(real) * nnz));

    CUDA_RT_CALL(cudaMemcpy(device_csrRowIndices, host_csrRowIndices, sizeof(int) * (num_rows + 1),
                            cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(device_csrColIndices, host_csrColIndices, sizeof(int) * nnz,
                            cudaMemcpyHostToDevice));
    CUDA_RT_CALL(
        cudaMemcpy(device_csrVal, host_csrVal, sizeof(real) * nnz, cudaMemcpyHostToDevice));

    // Set symmetric heap size for nvshmem based on problem size
    // Its default value in nvshmem is 1 GB which is not sufficient
    // for large mesh sizes
    long long unsigned int mesh_size_per_rank = num_rows / size + (num_rows % size != 0);

    long long unsigned int required_symmetric_heap_size =
        5 * mesh_size_per_rank * sizeof(real) * 1.1;

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

    if (compare_to_single_gpu) {
        CUDA_RT_CALL(cudaMallocHost(&x_ref_single_gpu, num_rows * sizeof(real)));

        single_gpu_runtime = SingleGPUDiscreteStandard::run_single_gpu(
            iter_max, device_csrRowIndices, device_csrColIndices, device_csrVal, x_ref_single_gpu,
            num_rows, nnz, matrix_is_zero_indexed);

        // single_gpu_runtime = SingleGPUDiscretePipelined::run_single_gpu(
        //     iter_max, device_csrRowIndices, device_csrColIndices, device_csrVal,
        //     x_ref_single_gpu, num_rows, nnz, matrix_is_zero_indexed);
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

        CPU::cpuConjugateGrad(iter_max, host_csrRowIndices, host_csrColIndices, host_csrVal,
                              x_ref_cpu, s_cpu, p_cpu, r_cpu, nnz, num_rows, tol);
    }

    CUDA_RT_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    int sMemSize = sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1);
    int numBlocks = (chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    nvshmem_barrier_all();

    double start = MPI_Wtime();

    NVSHMEM::initVectors<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
        device_r, device_x, row_start_global_idx, chunk_size, num_rows);

    nvshmemx_barrier_all_on_stream(mainStream);

    // ax0 = Ax0
    NVSHMEM::gpuSpMV<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
        device_csrRowIndices, device_csrColIndices, device_csrVal, real_positive_one, device_x,
        device_ax0, row_start_global_idx, chunk_size, num_rows, matrix_is_zero_indexed);

    nvshmemx_barrier_all_on_stream(mainStream);

    // r0 = b0 - ax0
    // NOTE: b is a unit vector.
    NVSHMEM::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
        device_ax0, device_r, real_negative_one, chunk_size);

    // // p0 = r0
    NVSHMEM::gpuCopyVector<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(device_r, device_p,
                                                                            chunk_size);

    // Do we need to reset the local dot here?
    resetLocalDotProduct<<<1, 1, 0, mainStream>>>(device_dot_gamma1);

    gpuDotProduct<<<numBlocks, THREADS_PER_BLOCK, sMemSize, mainStream>>>(
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
        NVSHMEM::gpuSpMV<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
            device_csrRowIndices, device_csrColIndices, device_csrVal, real_positive_one, device_p,
            device_s, row_start_global_idx, chunk_size, num_rows, matrix_is_zero_indexed);

        nvshmemx_barrier_all_on_stream(mainStream);

        resetLocalDotProduct<<<1, 1, 0, mainStream>>>(device_dot_delta1);

        gpuDotProduct<<<numBlocks, THREADS_PER_BLOCK, sMemSize, mainStream>>>(
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
        NVSHMEM::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(device_p, device_x,
                                                                           alpha, chunk_size);

        negative_alpha = -alpha;

        // r_(k+1) = r_k - alpha_k * s
        NVSHMEM::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
            device_s, device_r, negative_alpha, chunk_size);

        resetLocalDotProduct<<<1, 1, 0, mainStream>>>(device_dot_gamma1);

        gpuDotProduct<<<numBlocks, THREADS_PER_BLOCK, sMemSize, mainStream>>>(
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
        NVSHMEM::gpuScaleVectorAndSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, mainStream>>>(
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
    nvshmem_free(device_ax0);

    nvshmem_free(device_dot_delta1);
    nvshmem_free(device_dot_gamma1);

    CUDA_RT_CALL(cudaStreamDestroy(mainStream));

    CUDA_RT_CALL(cudaFree(device_csrRowIndices));
    CUDA_RT_CALL(cudaFree(device_csrColIndices));
    CUDA_RT_CALL(cudaFree(device_csrVal));

    free(host_csrRowIndices);
    free(host_csrColIndices);
    free(host_csrVal);

    if (compare_to_single_gpu || compare_to_cpu) {
        cudaFreeHost(x_final_result);

        if (compare_to_single_gpu) {
            cudaFreeHost(x_ref_single_gpu);
        }

        if (compare_to_cpu) {
            cudaFreeHost(x_ref_cpu);

            // Free CPU arrays here
        }
    }

    nvshmem_finalize();
    MPI_CALL(MPI_Finalize());

    return 0;
}