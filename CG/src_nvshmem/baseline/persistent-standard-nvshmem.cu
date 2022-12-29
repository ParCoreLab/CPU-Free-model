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

#include "../../include_nvshmem/baseline/persistent-standard-nvshmem.cuh"
#include "../../include_nvshmem/common.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

namespace cg = cooperative_groups;

namespace BaselinePersistentStandardNVSHMEM {

__device__ void gpuSpMV(int *I, int *J, real *val, real alpha, real *inputVecX, real *outputVecY,
                        int row_start_idx, int chunk_size, int num_rows,
                        bool matrix_is_zero_indexed) {
    int grid_rank = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;

    int mype = nvshmem_my_pe();

    for (int local_row_idx = grid_rank; local_row_idx < chunk_size; local_row_idx += grid_size) {
        int global_row_idx = row_start_idx + local_row_idx;

        if (global_row_idx < num_rows) {
            int row_elem = I[global_row_idx] - int(!matrix_is_zero_indexed);
            int next_row_elem = I[global_row_idx + 1] - int(!matrix_is_zero_indexed);
            int num_elems_this_row = next_row_elem - row_elem;

            real output = 0.0;

            for (int j = 0; j < num_elems_this_row; j++) {
                // If matrix is 1-indexed, need to move indices back by 1
                int input_vec_elem_idx = J[row_elem + j] - int(!matrix_is_zero_indexed);
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
                              const cg::thread_block &cta, int chunk_size) {
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

__device__ void gpuCopyVector(real *srcA, real *destB, int chunk_size) {
    int grid_rank = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;

    for (int i = grid_rank; i < chunk_size; i += grid_size) {
        destB[i] = srcA[i];
    }
}

__device__ void gpuSaxpy(real *x, real *y, real a, int chunk_size) {
    int grid_rank = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;

    for (int i = grid_rank; i < chunk_size; i += grid_size) {
        y[i] = a * x[i] + y[i];
    }
}

__device__ void gpuScaleVectorAndSaxpy(real *x, real *y, real a, real scale, int chunk_size) {
    int grid_rank = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;

    for (int i = grid_rank; i < chunk_size; i += grid_size) {
        y[i] = a * x[i] + scale * y[i];
    }
}

__global__ void __launch_bounds__(1024, 1)
    multiGpuConjugateGradient(int *I, int *J, real *val, real *x, real *r, real *p, real *s,
                              real *ax0, double *dot_delta1, double *dot_gamma1, int nnz,
                              int num_rows, int row_start_idx, int chunk_size,
                              bool matrix_is_zero_indexed, real tol, const int iter_max,
                              const int sMemSize) {
    int grid_rank = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;

    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    real real_positive_one = 1.0;
    real real_negative_one = -1.0;

    real tmp_dot_gamma0 = 0.0;

    real beta;
    real alpha;
    real negative_alpha;

    int mype = nvshmem_my_pe();

    for (int local_row_idx = grid_rank; local_row_idx < chunk_size; local_row_idx += grid_size) {
        int global_row_idx = row_start_idx + local_row_idx;

        if (global_row_idx < num_rows) {
            r[local_row_idx] = 1.0;
            x[local_row_idx] = 0.0;
        }
    }

    if (grid.thread_rank() == 0) {
        nvshmem_barrier_all();
    }

    cg::sync(grid);

    // ax0 = AX0
    gpuSpMV(I, J, val, real_positive_one, x, ax0, row_start_idx, chunk_size, num_rows,
            matrix_is_zero_indexed);

    if (grid.thread_rank() == 0) {
        nvshmem_barrier_all();
    }

    cg::sync(grid);

    // r0 = b0 - ax0
    // NOTE: b is a unit vector.
    gpuSaxpy(ax0, r, real_negative_one, chunk_size);

    if (grid.thread_rank() == 0) {
        nvshmem_barrier_all();
    }

    cg::sync(grid);

    // p0 = r0
    gpuCopyVector(r, p, chunk_size);

    cg::sync(grid);

    if (grid.thread_rank() == 0) {
        *dot_gamma1 = 0.0;
    }

    cg::sync(grid);

    gpuDotProduct(r, r, dot_gamma1, cta, chunk_size);

    cg::sync(grid);

    if (grid.thread_rank() == 0) {
        nvshmem_double_sum_reduce(NVSHMEM_TEAM_WORLD, dot_gamma1, dot_gamma1, 1);
        nvshmem_barrier_all();
    }

    cg::sync(grid);

    tmp_dot_gamma0 = (real)*dot_gamma1;

    int k = 1;

    while (k <= iter_max) {
        gpuSpMV(I, J, val, real_positive_one, p, s, row_start_idx, chunk_size, num_rows,
                matrix_is_zero_indexed);

        if (grid.thread_rank() == 0) {
            nvshmem_barrier_all();
        }

        cg::sync(grid);

        if (grid.thread_rank() == 0) {
            *dot_delta1 = 0.0;
        }

        cg::sync(grid);

        gpuDotProduct(p, s, dot_delta1, cta, chunk_size);

        cg::sync(grid);

        if (grid.thread_rank() == 0) {
            nvshmem_double_sum_reduce(NVSHMEM_TEAM_WORLD, dot_delta1, dot_delta1, 1);
            nvshmem_barrier_all();
        }

        cg::sync(grid);

        alpha = tmp_dot_gamma0 / (real)*dot_delta1;

        gpuSaxpy(p, x, alpha, chunk_size);

        negative_alpha = -alpha;

        gpuSaxpy(s, r, negative_alpha, chunk_size);

        if (grid.thread_rank() == 0) {
            *dot_gamma1 = 0.0;
        }

        cg::sync(grid);

        gpuDotProduct(r, r, dot_gamma1, cta, chunk_size);

        cg::sync(grid);

        if (grid.thread_rank() == 0) {
            nvshmem_double_sum_reduce(NVSHMEM_TEAM_WORLD, dot_gamma1, dot_gamma1, 1);
            nvshmem_barrier_all();
        }

        cg::sync(grid);

        real tmp_dot_gamma1 = (real)*dot_gamma1;

        beta = tmp_dot_gamma1 / tmp_dot_gamma0;

        gpuScaleVectorAndSaxpy(r, p, real_positive_one, beta, chunk_size);

        tmp_dot_gamma0 = tmp_dot_gamma1;

        if (grid.thread_rank() == 0) {
            nvshmem_barrier_all();
        }

        cg::sync(grid);

        k++;
    }
}
}  // namespace BaselinePersistentStandardNVSHMEM

int BaselinePersistentStandardNVSHMEM::init(int argc, char *argv[]) {
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
    real *device_ax0;

    double *device_dot_delta1;
    double *device_dot_gamma1;

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

    // Check if matrix is 0 or 1 indexed
    int index_base = host_I[0];

    if (index_base == 1) {
        matrix_is_zero_indexed = false;
    } else if (index_base == 0) {
        matrix_is_zero_indexed = true;
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
            iter_max, device_I, device_J, device_val, x_ref_single_gpu, num_rows, nnz,
            matrix_is_zero_indexed);

        // single_gpu_runtime = SingleGPUDiscretePipelined::run_single_gpu(
        //     iter_max, device_I, device_J, device_val, x_ref_single_gpu, num_rows, nnz,
        //     matrix_is_zero_indexed);
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

    // WARNING!!!
    // This was causing issues for me
    // Get rid of THREADS_PER_BLOCK
    // Use per version threadsPerBlock variable
    int threadsPerBlock = 1024;
    int sMemSize = sizeof(double) * ((threadsPerBlock / 32) + 1);

    void *kernelArgs[] = {
        (void *)&device_I,
        (void *)&device_J,
        (void *)&device_val,
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

    nvshmem_barrier_all();
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
