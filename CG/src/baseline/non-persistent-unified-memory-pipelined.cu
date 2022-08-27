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

#include "../../include/baseline/non-persistent-unified-memory-pipelined.cuh"
#include "../../include/common.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace BaselineNonPersistentUnifiedMemoryPipelined {

#define ENABLE_CPU_DEBUG_CODE 0
#define THREADS_PER_BLOCK 512

__device__ double grid_dot_result = 0.0;

__global__ void initVectors(float *rhs, float *x, int num_rows, const int device_rank,
                            const int num_devices) {
    size_t local_grid_size = gridDim.x * blockDim.x;
    size_t local_grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    size_t global_grid_size = local_grid_size * num_devices;
    size_t global_grid_rank = device_rank * local_grid_size + local_grid_rank;

    for (size_t i = global_grid_rank; i < num_rows; i += global_grid_size) {
        rhs[i] = 1.0;
        x[i] = 0.0;
    }
}

__global__ void gpuSpMV(int *I, int *J, float *val, int nnz, int num_rows, float alpha,
                        float *inputVecX, float *outputVecY, const int device_rank,
                        const int num_devices) {
    size_t local_grid_size = gridDim.x * blockDim.x;
    size_t local_grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    size_t global_grid_size = local_grid_size * num_devices;
    size_t global_grid_rank = device_rank * local_grid_size + local_grid_rank;

    for (size_t i = global_grid_rank; i < num_rows; i += global_grid_size) {
        int row_elem = I[i];
        int next_row_elem = I[i + 1];
        int num_elems_this_row = next_row_elem - row_elem;

        float output = 0.0;
        for (int j = 0; j < num_elems_this_row; j++) {
            output += alpha * val[row_elem + j] * inputVecX[J[row_elem + j]];
        }

        outputVecY[i] = output;
    }
}

__global__ void gpuSaxpy(float *x, float *y, float a, int size, const int device_rank,
                         const int num_devices) {
    size_t local_grid_size = gridDim.x * blockDim.x;
    size_t local_grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    size_t global_grid_size = local_grid_size * num_devices;
    size_t global_grid_rank = device_rank * local_grid_size + local_grid_rank;

    for (size_t i = global_grid_rank; i < num_rows; i += global_grid_size) {
        y[i] = a * x[i] + y[i];
    }
}

__global__ void gpuDotProduct(float *vecA, float *vecB, int num_rows, const int device_rank,
                              const int num_devices) {
    cg::thread_block cta = cg::this_thread_block();

    size_t local_grid_size = gridDim.x * blockDim.x;
    size_t local_grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    size_t global_grid_size = local_grid_size * num_devices;
    size_t global_grid_rank = device_rank * local_grid_size + local_grid_rank;

    extern __shared__ double tmp[];

    double temp_sum = 0.0;

    for (size_t i = global_grid_rank; i < num_rows; i += global_grid_size) {
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
            atomicAdd(&grid_dot_result, temp_sum);
        }
    }
}

__global__ void gpuCopyVector(float *srcA, float *destB, int size, const int device_rank,
                              const int num_devices) {
    size_t local_grid_size = gridDim.x * blockDim.x;
    size_t local_grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    size_t global_grid_size = local_grid_size * num_devices;
    size_t global_grid_rank = device_rank * local_grid_size + local_grid_rank;

    for (size_t i = global_grid_rank; i < num_rows; i += global_grid_size) {
        destB[i] = srcA[i];
    }
}

__global__ void gpuScaleVectorAndSaxpy(float *x, float *y, float a, float scale, int size,
                                       const int device_rank, const int num_devices) {
    size_t local_grid_size = gridDim.x * blockDim.x;
    size_t local_grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    size_t global_grid_size = local_grid_size * num_devices;
    size_t global_grid_rank = device_rank * local_grid_size + local_grid_rank;

    for (size_t i = global_grid_rank; i < num_rows; i += global_grid_size) {
        y[i] = a * x[i] + scale * y[i];
    }
}
}  // namespace BaselineNonPersistentUnifiedMemoryPipelined

int BaselineNonPersistentUnifiedMemoryPipelined::init(int argc, char *argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 10000);
    std::string matrix_path_str = get_argval<std::string>(argv, argv + argc, "-matrix_path", "");
    const bool compare_to_cpu = get_arg(argv, argv + argc, "-compare");

    char *matrix_path_char = const_cast<char *>(matrix_path_str.c_str());
    bool generate_random_tridiag_matrix = matrix_path_str.empty();

    std::string matrix_name = std::filesystem::path(matrix_path_str).stem();

    if (generate_random_tridiag_matrix) {
        matrix_name = "random tridiagonal";
    }

    // std::cout << "Running on matrix: " << matrix_name << "\n" << std::endl;

    int num_devices = 0;

    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

    int num_rows = 0;
    int num_cols = 0;
    int nnz = 0;

    int *host_I = NULL;
    int *host_J = NULL;
    float *host_val = NULL;

    int *um_I = NULL;
    int *um_J = NULL;
    float *um_val = NULL;

    const float tol = 1e-5f;
    float *x;
    float rhs = 1.0;
    float r1;
    float *r, *p, *Ax;

    cudaStream_t streamDefault[num_devices];
    cudaStream_t streamSaxpy[num_devices];
    cudaStream_t streamDot[num_devices];
    cudaStream_t streamSpMV[num_devices];

    for (int gpu_idx_i = 0; gpu_idx_i < num_devices; gpu_idx_i++) {
        CUDA_RT_CALL(cudaSetDevice(gpu_idx_i));

        for (int gpu_idx_j = 0; gpu_idx_j < num_devices; gpu_idx_j++) {
            if (gpu_idx_i != gpu_idx_j) {
                CUDA_RT_CALL(cudaDeviceEnablePeerAccess(gpu_idx_j, 0));
                CUDA_RT_CALL(cudaSetDevice(gpu_idx_i));
            }
        }
    }

    if (generate_random_tridiag_matrix) {
        num_rows = 10485760 * 2;
        num_cols = num_rows;

        nnz = (num_rows - 2) * 3 + 4;

        CUDA_RT_CALL(cudaMallocManaged((void **)&um_I, sizeof(int) * (num_rows + 1)));
        CUDA_RT_CALL(cudaMallocManaged((void **)&um_J, sizeof(int) * nnz));
        CUDA_RT_CALL(cudaMallocManaged((void **)&um_val, sizeof(float) * nnz));

        host_val = (float *)malloc(sizeof(float) * nnz);

        /* Generate a random tridiagonal symmetric matrix in CSR format */
        genTridiag(um_I, um_J, host_val, num_rows, nnz);

        memcpy(um_val, host_val, sizeof(float) * nnz);

    } else {
        if (loadMMSparseMatrix<float>(matrix_path_char, 'd', true, &num_rows, &num_cols, &nnz,
                                      &host_val, &host_I, &host_J, true)) {
            exit(EXIT_FAILURE);
        }

        CUDA_RT_CALL(cudaMallocManaged((void **)&um_I, sizeof(int) * (num_rows + 1)));
        CUDA_RT_CALL(cudaMallocManaged((void **)&um_J, sizeof(int) * nnz));
        CUDA_RT_CALL(cudaMallocManaged((void **)&um_val, sizeof(float) * nnz));

        memcpy(um_I, host_I, sizeof(int) * (num_rows + 1));
        memcpy(um_J, host_J, sizeof(int) * nnz);
        memcpy(um_val, host_val, sizeof(float) * nnz);
    }

    CUDA_RT_CALL(cudaMallocManaged((void **)&x, sizeof(float) * num_rows));

    double *dot_result;
    CUDA_RT_CALL(cudaMallocManaged((void **)&dot_result, sizeof(double)));

    CUDA_RT_CALL(cudaMemset(dot_result, 0, sizeof(double)));

    // temp memory for ConjugateGradient
    CUDA_RT_CALL(cudaMallocManaged((void **)&r, num_rows * sizeof(float)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&p, num_rows * sizeof(float)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&Ax, num_rows * sizeof(float)));

    float *d_r1, *d_r0, *d_dot, *d_a, *d_na, *d_b;
    checkCudaErrors(cudaMallocManaged((void **)&d_r1, sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&d_r0, sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&d_dot, sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&d_a, sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&d_na, sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&d_b, sizeof(float)));

    // ASSUMPTION: All GPUs are the same and P2P callable

    cudaStream_t nStreams[num_devices];

    int sMemSize = sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1);
    int numThreads = THREADS_PER_BLOCK;

#if ENABLE_CPU_DEBUG_CODE
    float *Ax_cpu = (float *)malloc(sizeof(float) * N);
    float *r_cpu = (float *)malloc(sizeof(float) * N);
    float *p_cpu = (float *)malloc(sizeof(float) * N);
    float *x_cpu = (float *)malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        r_cpu[i] = 1.0;
        Ax_cpu[i] = x_cpu[i] = 0.0;
    }
#endif

    dim3 dimGrid(numSms * numBlocksPerSm, 1, 1), dimBlock(numThreads, 1, 1);

    double start = omp_get_wtime();

    float alpha = 1.0;
    float alpham1 = -1.0;
    float beta = 0.0;

    int numBlocksInitVectors = 0;
    int numBlocksSpmv = 0;
    int numBlocksSaxpy = 0;
    int numBlocksDotProduct = 0;

    int blockSizeInitVectors = 0;

    checkCudaErrors(
        cudaOccupancyMaxPotentialBlockSize(&numBlocksInitVectors, &blockSize, initVectors));
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&numBlocksSpmv, THREADS_PER_BLOCK, gpuSpMV));
    checkCudaErrors(
        cudaOccupancyMaxPotentialBlockSize(&numBlocksSaxpy, THREADS_PER_BLOCK, gpuSaxpy));
    checkCudaErrors(
        cudaOccupancyMaxPotentialBlockSize(&numBlocksDotProduct, THREADS_PER_BLOCK, gpuDotProduct));

    for (int gpu_idx = 0; gpu_idx < num_devices; gpu_idx++) {
        initVectors<<<numBlocksInitVectors, blockSize, 0, stream1>>>(r, x, N, gpu_idx, num_devices);

        gpuSpMV<<<numBlocks, THREADS_PER_BLOCK, 0, streamDefault>>>(
            um_I, um_J, um_val, nnz, num_rows, alpha, x, Ax, gpu_idx, num_devices);

        gpuSaxpy<<<numBlocksSaxpy, THREADS_PER_BLOCK, 0, streamDefault>>>(Ax, r, alpham1, num_rows,
                                                                          gpu_idx, num_devices);

        gpuDotProduct<<<numBlocksDotProduct, THREADS_PER_BLOCK, 0, streamDefault>>>(
            r, r, num_rows, gpu_idx, num_devices);
    }
}

// for (int gpu_idx = 0; gpu_idx < num_devices; gpu_idx++) {
//     CUDA_RT_CALL(cudaSetDevice(gpu_idx));
//     CUDA_RT_CALL(cudaStreamSynchronize(nStreams[gpu_idx]));
// }

r1 = (float)*dot_result;

double stop = omp_get_wtime();

printf("Execution time: %8.4f s\n", (stop - start));

#if ENABLE_CPU_DEBUG_CODE
cpuConjugateGrad(I, J, val, x_cpu, Ax_cpu, p_cpu, r_cpu, nz, N, tol);
#endif

float rsum, diff, err = 0.0;

for (int i = 0; i < num_rows; i++) {
    rsum = 0.0;

    for (int j = um_I[i]; j < um_J[i + 1]; j++) {
        rsum += host_val[j] * x[um_J[j]];
    }

    diff = fabs(rsum - rhs);

    if (diff > err) {
        err = diff;
    }
}

CUDA_RT_CALL(cudaFreeHost(multi_device_data.hostMemoryArrivedList));
CUDA_RT_CALL(cudaFree(um_I));
CUDA_RT_CALL(cudaFree(um_J));
CUDA_RT_CALL(cudaFree(um_val));
CUDA_RT_CALL(cudaFree(x));
CUDA_RT_CALL(cudaFree(r));
CUDA_RT_CALL(cudaFree(p));
CUDA_RT_CALL(cudaFree(Ax));
CUDA_RT_CALL(cudaFree(dot_result));
free(host_val);

#if ENABLE_CPU_DEBUG_CODE
free(Ax_cpu);
free(r_cpu);
free(p_cpu);
free(x_cpu);
#endif

if (compare_to_cpu) {
    printf("GPU Final, residual = %e \n  ", sqrt(r1));
    printf("Test Summary:  Error amount = %f \n", err);
    fprintf(stdout, "&&&& conjugateGradientMultiDeviceCG %s\n",
            (sqrt(r1) < tol) ? "PASSED" : "FAILED");
}

return 0;
}