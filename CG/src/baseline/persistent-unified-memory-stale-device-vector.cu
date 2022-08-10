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

#include <curand.h>

#include <omp.h>

#include "../../include/baseline/persistent-unified-memory-stale-device-vector.cuh"
#include "../../include/common.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace BaselinePersistentUnifiedMemoryStaleDeviceVector {

#define ENABLE_CPU_DEBUG_CODE 0
#define THREADS_PER_BLOCK 512

__device__ double grid_dot_result = 0.0;

__global__ void multiGpuConjugateGradient(int *I, int *J, float *val, float *x, float *Ax,
                                          float *um_p, float *device_p, float *r,
                                          double *dot_result, int nnz, int N, float tol,
                                          MultiDeviceData multi_device_data, const int iter_max) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    PeerGroup peer_group(multi_device_data, grid);

    float alpha = 1.0;
    float alpham1 = -1.0;
    float r0 = 0.0, r1, b, a, na;

    for (int i = peer_group.thread_rank(); i < N; i += peer_group.size()) {
        r[i] = 1.0;
        x[i] = 0.0;
    }

    cg::sync(grid);

    gpuSpMV(I, J, val, nnz, N, alpha, x, Ax, peer_group);

    cg::sync(grid);

    gpuSaxpy(Ax, r, alpham1, N, peer_group);

    cg::sync(grid);

    gpuDotProduct(r, r, N, cta, peer_group, &grid_dot_result);

    cg::sync(grid);

    if (grid.thread_rank() == 0) {
        atomicAdd_system(dot_result, grid_dot_result);
        grid_dot_result = 0.0;
    }

    peer_group.sync();

    r1 = *dot_result;

    int k = 1;

    // Full CG
    while (k <= iter_max) {
        // Saxpy 1 Start

        if (k > 1) {
            b = r1 / r0;
            gpuScaleVectorAndSaxpy(r, um_p, alpha, b, N, peer_group);
        } else {
            gpuCopyVector(r, um_p, N, peer_group);
        }

        peer_group.sync();

        // Saxpy 1 End

        // SpMV Start

        gpuSpMV(I, J, val, nnz, N, alpha, device_p, Ax, peer_group);

        // SpMV End

        // Dot Product 1 Start

        if (peer_group.thread_rank() == 0) {
            *dot_result = 0.0;
        }
        peer_group.sync();

        gpuDotProduct(um_p, Ax, N, cta, peer_group, &grid_dot_result);

        cg::sync(grid);

        if (grid.thread_rank() == 0) {
            atomicAdd_system(dot_result, grid_dot_result);
            grid_dot_result = 0.0;
        }

        peer_group.sync();

        // Dot Product 1 End

        // Saxpy 2 Start

        a = r1 / *dot_result;

        gpuSaxpy(um_p, x, a, N, peer_group);

        na = -a;

        gpuSaxpy(Ax, r, na, N, peer_group);

        r0 = r1;

        peer_group.sync();

        // Saxpy 2 End

        // Dot Product 2 Start

        if (peer_group.thread_rank() == 0) {
            *dot_result = 0.0;
        }

        peer_group.sync();

        gpuDotProduct(r, r, N, cta, peer_group, &grid_dot_result);

        cg::sync(grid);

        if (grid.thread_rank() == 0) {
            atomicAdd_system(dot_result, grid_dot_result);
            grid_dot_result = 0.0;
        }
        peer_group.sync();

        // Dot Product 2 End

        // Saxpy 3 Start

        r1 = *dot_result;

        // Saxpy 3 End

        k++;
    }

    // // Saxpy
    // while (k <= iter_max) {
    //     // Saxpy 1 Start

    //     if (k > 1) {
    //         b = r1 / r0;
    //         gpuScaleVectorAndSaxpy(r, um_p, alpha, b, N, peer_group);
    //     } else {
    //         gpuCopyVector(r, um_p, N, peer_group);
    //     }

    //     peer_group.sync();

    //     // Saxpy 2 Start

    //     a = r1 / *dot_result;

    //     gpuSaxpy(um_p, x, a, N, peer_group);

    //     na = -a;

    //     gpuSaxpy(Ax, r, na, N, peer_group);

    //     r0 = r1;

    //     peer_group.sync();

    //     // Saxpy 3 Start

    //     r1 = *dot_result;

    //     // Saxpy 3 End

    //     k++;
    // }

    // // SpMV
    // while (k <= iter_max) {
    //     // SpMV Start

    //     gpuSpMV(I, J, val, nnz, N, alpha, device_p, Ax, peer_group);

    //     k++;
    // }

    // // Dot
    // while (k <= iter_max) {
    //     // Dot Product 1 Start

    //     if (peer_group.thread_rank() == 0) {
    //         *dot_result = 0.0;
    //     }
    //     peer_group.sync();

    //     gpuDotProduct(um_p, Ax, N, cta, peer_group, &grid_dot_result);

    //     cg::sync(grid);

    //     if (grid.thread_rank() == 0) {
    //         atomicAdd_system(dot_result, grid_dot_result);
    //         grid_dot_result = 0.0;
    //     }

    //     peer_group.sync();

    //     // Dot Product 2 Start

    //     if (peer_group.thread_rank() == 0) {
    //         *dot_result = 0.0;
    //     }

    //     peer_group.sync();

    //     gpuDotProduct(r, r, N, cta, peer_group, &grid_dot_result);

    //     cg::sync(grid);

    //     if (grid.thread_rank() == 0) {
    //         atomicAdd_system(dot_result, grid_dot_result);
    //         grid_dot_result = 0.0;
    //     }

    //     peer_group.sync();

    //     k++;
    // }
}
}  // namespace BaselinePersistentUnifiedMemoryStaleDeviceVector

int BaselinePersistentUnifiedMemoryStaleDeviceVector::init(int argc, char *argv[]) {
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

    curandGenerator_t gen;

    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

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
    float *r, *Ax;

    float *um_p;
    float *device_p;

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

    CUDA_RT_CALL(cudaMemAdvise(um_I, sizeof(int) * (num_rows + 1), cudaMemAdviseSetReadMostly, 0));
    CUDA_RT_CALL(cudaMemAdvise(um_J, sizeof(int) * nnz, cudaMemAdviseSetReadMostly, 0));
    CUDA_RT_CALL(cudaMemAdvise(um_val, sizeof(float) * nnz, cudaMemAdviseSetReadMostly, 0));

    CUDA_RT_CALL(cudaMallocManaged((void **)&x, sizeof(float) * num_rows));

    double *dot_result;
    CUDA_RT_CALL(cudaMallocManaged((void **)&dot_result, sizeof(double)));

    CUDA_RT_CALL(cudaMemset(dot_result, 0, sizeof(double)));

    // temp memory for ConjugateGradient
    CUDA_RT_CALL(cudaMallocManaged((void **)&r, num_rows * sizeof(float)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_p, num_rows * sizeof(float)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&Ax, num_rows * sizeof(float)));
    CUDA_RT_CALL(cudaMalloc((void **)&device_p, num_rows * sizeof(float)));

    CURAND_CALL(curandGenerateUniform(gen, device_p, num_rows));

    // ASSUMPTION: All GPUs are the same and P2P callable

    cudaStream_t nStreams[num_devices];

    int sMemSize = sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1);
    int numBlocksPerSm = INT_MAX;
    int numThreads = THREADS_PER_BLOCK;

    CUDA_RT_CALL(cudaSetDevice(0));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int numSms = deviceProp.multiProcessorCount;

    // Added this line to time the different kernel operations
    numBlocksPerSm = 2;

    if (!numBlocksPerSm) {
        printf(
            "Max active blocks per SM is returned as 0.\n Hence, Waiving the "
            "sample\n");
    }

    int totalThreadsPerGPU = numSms * numBlocksPerSm * numThreads;

    for (int gpu_idx = 0; gpu_idx < num_devices; gpu_idx++) {
        CUDA_RT_CALL(cudaSetDevice(gpu_idx));
        CUDA_RT_CALL(cudaStreamCreate(&nStreams[gpu_idx]));

        int perGPUIter = num_rows / (totalThreadsPerGPU * num_devices);
        int offset_Ax = gpu_idx * totalThreadsPerGPU;
        int offset_r = gpu_idx * totalThreadsPerGPU;
        int offset_p = gpu_idx * totalThreadsPerGPU;
        int offset_x = gpu_idx * totalThreadsPerGPU;

        CUDA_RT_CALL(
            cudaMemPrefetchAsync(um_I, sizeof(int) * num_rows, gpu_idx, nStreams[gpu_idx]));
        CUDA_RT_CALL(cudaMemPrefetchAsync(um_val, sizeof(float) * nnz, gpu_idx, nStreams[gpu_idx]));
        CUDA_RT_CALL(cudaMemPrefetchAsync(um_J, sizeof(float) * nnz, gpu_idx, nStreams[gpu_idx]));

        if (offset_Ax <= num_rows) {
            for (int i = 0; i < perGPUIter; i++) {
                cudaMemAdvise(Ax + offset_Ax, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetPreferredLocation, gpu_idx);
                cudaMemAdvise(r + offset_r, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetPreferredLocation, gpu_idx);
                cudaMemAdvise(x + offset_x, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetPreferredLocation, gpu_idx);
                cudaMemAdvise(um_p + offset_p, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetPreferredLocation, gpu_idx);

                cudaMemAdvise(Ax + offset_Ax, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetAccessedBy, gpu_idx);
                cudaMemAdvise(r + offset_r, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetAccessedBy, gpu_idx);
                cudaMemAdvise(um_p + offset_p, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetAccessedBy, gpu_idx);
                cudaMemAdvise(x + offset_x, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetAccessedBy, gpu_idx);

                offset_Ax += totalThreadsPerGPU * num_devices;
                offset_r += totalThreadsPerGPU * num_devices;
                offset_p += totalThreadsPerGPU * num_devices;
                offset_x += totalThreadsPerGPU * num_devices;

                if (offset_Ax >= num_rows) {
                    break;
                }
            }
        }
    }

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

    // Structure used for cross-grid synchronization.
    MultiDeviceData multi_device_data;
    CUDA_RT_CALL(cudaHostAlloc(&multi_device_data.hostMemoryArrivedList,
                               (num_devices - 1) * sizeof(*multi_device_data.hostMemoryArrivedList),
                               cudaHostAllocPortable));
    memset(multi_device_data.hostMemoryArrivedList, 0,
           (num_devices - 1) * sizeof(*multi_device_data.hostMemoryArrivedList));
    multi_device_data.numDevices = num_devices;
    multi_device_data.deviceRank = 0;

    void *kernelArgs[] = {
        (void *)&um_I,
        (void *)&um_J,
        (void *)&um_val,
        (void *)&x,
        (void *)&Ax,
        (void *)&um_p,
        (void *)&device_p,
        (void *)&r,
        (void *)&dot_result,
        (void *)&nnz,
        (void *)&num_rows,
        (void *)&tol,
        (void *)&multi_device_data,
        (void *)&iter_max,
    };

    double start = omp_get_wtime();

    for (int gpu_idx = 0; gpu_idx < num_devices; gpu_idx++) {
        CUDA_RT_CALL(cudaSetDevice(gpu_idx));
        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *)multiGpuConjugateGradient, dimGrid,
                                                 dimBlock, kernelArgs, sMemSize,
                                                 nStreams[gpu_idx]));
        multi_device_data.deviceRank++;
    }

    CUDA_RT_CALL(cudaMemPrefetchAsync(x, sizeof(float) * num_rows, cudaCpuDeviceId));
    CUDA_RT_CALL(cudaMemPrefetchAsync(dot_result, sizeof(double), cudaCpuDeviceId));

    for (int gpu_idx = 0; gpu_idx < num_devices; gpu_idx++) {
        CUDA_RT_CALL(cudaSetDevice(gpu_idx));
        CUDA_RT_CALL(cudaStreamSynchronize(nStreams[gpu_idx]));
    }

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
    CUDA_RT_CALL(cudaFree(um_p));
    CUDA_RT_CALL(cudaFree(device_p));
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