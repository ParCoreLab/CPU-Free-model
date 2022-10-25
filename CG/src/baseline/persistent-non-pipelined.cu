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

#include "../../include/baseline/persistent-non-pipelined.cuh"
#include "../../include/common.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace BaselinePersistentNonPipelined {

__device__ double grid_dot_result = 0.0;

__device__ void gpuSpMV(int *I, int *J, float *val, int nnz, int num_rows, float alpha,
                        float *inputVecX, float *outputVecY, const PeerGroup &peer_group) {
    for (int i = peer_group.thread_rank(); i < num_rows; i += peer_group.size()) {
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

__device__ void gpuSaxpy(float *x, float *y, float a, int size, const PeerGroup &peer_group) {
    for (int i = peer_group.thread_rank(); i < size; i += peer_group.size()) {
        y[i] = a * x[i] + y[i];
    }
}

__device__ void gpuDotProduct(float *vecA, float *vecB, int size, const cg::thread_block &cta,
                              const PeerGroup &peer_group) {
    extern __shared__ double tmp[];

    double temp_sum = 0.0;

    for (int i = peer_group.thread_rank(); i < size; i += peer_group.size()) {
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

__device__ void gpuCopyVector(float *srcA, float *destB, int size, const PeerGroup &peer_group) {
    for (int i = peer_group.thread_rank(); i < size; i += peer_group.size()) {
        destB[i] = srcA[i];
    }
}

__device__ void gpuScaleVectorAndSaxpy(float *x, float *y, float a, float scale, int size,
                                       const PeerGroup &peer_group) {
    for (int i = peer_group.thread_rank(); i < size; i += peer_group.size()) {
        y[i] = a * x[i] + scale * y[i];
    }
}

__global__ void multiGpuConjugateGradient(int *I, int *J, float *val, float *x, float *Ax, float *p,
                                          float *r, double *dot_result, int nnz, int N, float tol,
                                          MultiDeviceData multi_device_data, const int iter_max) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    PeerGroup peer_group(multi_device_data, grid);

    float float_positive_one = 1.0;
    float float_negative_one = -1.0;

    float r0 = 0.0;
    float r1;
    float b;
    float a;
    float na;

    for (int i = peer_group.thread_rank(); i < N; i += peer_group.size()) {
        r[i] = 1.0;
        x[i] = 0.0;
    }

    cg::sync(grid);

    gpuSpMV(I, J, val, nnz, N, float_positive_one, x, Ax, peer_group);

    cg::sync(grid);

    gpuSaxpy(Ax, r, float_negative_one, N, peer_group);

    cg::sync(grid);

    gpuDotProduct(r, r, N, cta, peer_group);

    cg::sync(grid);

    if (grid.thread_rank() == 0) {
        atomicAdd_system(dot_result, grid_dot_result);
        grid_dot_result = 0.0;
    }
    peer_group.sync();

    r1 = *dot_result;

    int k = 1;

    while (k <= iter_max) {
        if (k > 1) {
            b = r1 / r0;
            gpuScaleVectorAndSaxpy(r, p, float_positive_one, b, N, peer_group);
        } else {
            gpuCopyVector(r, p, N, peer_group);
        }

        peer_group.sync();

        gpuSpMV(I, J, val, nnz, N, float_positive_one, p, Ax, peer_group);

        if (peer_group.thread_rank() == 0) {
            *dot_result = 0.0;
        }
        peer_group.sync();

        gpuDotProduct(p, Ax, N, cta, peer_group);

        cg::sync(grid);

        if (grid.thread_rank() == 0) {
            atomicAdd_system(dot_result, grid_dot_result);
            grid_dot_result = 0.0;
        }

        peer_group.sync();

        a = r1 / *dot_result;

        gpuSaxpy(p, x, a, N, peer_group);

        na = -a;

        gpuSaxpy(Ax, r, na, N, peer_group);

        r0 = r1;

        peer_group.sync();

        if (peer_group.thread_rank() == 0) {
            *dot_result = 0.0;
        }

        peer_group.sync();

        gpuDotProduct(r, r, N, cta, peer_group);

        cg::sync(grid);

        if (grid.thread_rank() == 0) {
            atomicAdd_system(dot_result, grid_dot_result);
            grid_dot_result = 0.0;
        }
        peer_group.sync();

        r1 = *dot_result;

        k++;
    }
}
}  // namespace BaselinePersistentNonPipelined

int BaselinePersistentNonPipelined::init(int argc, char *argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 10000);
    std::string matrix_path_str = get_argval<std::string>(argv, argv + argc, "-matrix_path", "");
    const bool compare_to_single_gpu = get_arg(argv, argv + argc, "-compare-single-gpu");

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
    float *host_val = NULL;
    float *x_host = NULL;
    float *x_ref_host = NULL;

    int *um_I = NULL;
    int *um_J = NULL;
    float *um_val = NULL;

    float r1;

    float *um_x;
    float *um_r;
    float *um_p;
    float *um_s;

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

    CUDA_RT_CALL(cudaMallocManaged((void **)&um_x, sizeof(float) * num_rows));

    // Comparing to Single GPU Non-Persistent Non-Pipelined implementation
#pragma omp parallel num_threads(num_devices)
    {
        int gpu_idx = omp_get_thread_num();

        if (compare_to_single_gpu && gpu_idx == 0) {
            CUDA_RT_CALL(cudaSetDevice(gpu_idx));

            CUDA_RT_CALL(cudaMallocHost(&x_ref_host, num_rows * sizeof(float)));
            CUDA_RT_CALL(cudaMallocHost(&x_host, num_rows * sizeof(float)));

            single_gpu_runtime = SingleGPUPipelinedDiscrete::run_single_gpu(
                iter_max, matrix_path_char, generate_random_tridiag_matrix, um_I, um_J, um_val,
                x_ref_host, num_rows, nnz);

            // single_gpu_runtime = SingleGPUStandardDiscrete::run_single_gpu(
            //     iter_max, matrix_path_char, generate_random_tridiag_matrix, um_I, um_J, um_val,
            //     x_ref_host, num_rows, nnz);
        }
    }

    double *dot_result;
    CUDA_RT_CALL(cudaMallocManaged((void **)&dot_result, sizeof(double)));

    CUDA_RT_CALL(cudaMemset(dot_result, 0, sizeof(double)));

    // temp memory for ConjugateGradient
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_r, num_rows * sizeof(float)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_p, num_rows * sizeof(float)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_s, num_rows * sizeof(float)));

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
        int offset_s = gpu_idx * totalThreadsPerGPU;
        int offset_r = gpu_idx * totalThreadsPerGPU;
        int offset_p = gpu_idx * totalThreadsPerGPU;
        int offset_x = gpu_idx * totalThreadsPerGPU;

        CUDA_RT_CALL(
            cudaMemPrefetchAsync(um_I, sizeof(int) * num_rows, gpu_idx, nStreams[gpu_idx]));
        CUDA_RT_CALL(cudaMemPrefetchAsync(um_val, sizeof(float) * nnz, gpu_idx, nStreams[gpu_idx]));
        CUDA_RT_CALL(cudaMemPrefetchAsync(um_J, sizeof(float) * nnz, gpu_idx, nStreams[gpu_idx]));

        if (offset_s <= num_rows) {
            for (int i = 0; i < perGPUIter; i++) {
                cudaMemAdvise(um_s + offset_s, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetPreferredLocation, gpu_idx);
                cudaMemAdvise(um_r + offset_r, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetPreferredLocation, gpu_idx);
                cudaMemAdvise(um_x + offset_x, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetPreferredLocation, gpu_idx);
                cudaMemAdvise(um_p + offset_p, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetPreferredLocation, gpu_idx);

                cudaMemAdvise(um_s + offset_s, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetAccessedBy, gpu_idx);
                cudaMemAdvise(um_r + offset_r, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetAccessedBy, gpu_idx);
                cudaMemAdvise(um_p + offset_p, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetAccessedBy, gpu_idx);
                cudaMemAdvise(um_x + offset_x, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetAccessedBy, gpu_idx);

                offset_s += totalThreadsPerGPU * num_devices;
                offset_r += totalThreadsPerGPU * num_devices;
                offset_p += totalThreadsPerGPU * num_devices;
                offset_x += totalThreadsPerGPU * num_devices;

                if (offset_s >= num_rows) {
                    break;
                }
            }
        }
    }

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
        (void *)&um_I,     (void *)&um_J,     (void *)&um_val, (void *)&um_x,
        (void *)&um_s,     (void *)&um_p,     (void *)&um_r,   (void *)&dot_result,
        (void *)&nnz,      (void *)&num_rows, (void *)&tol,    (void *)&multi_device_data,
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

    CUDA_RT_CALL(cudaMemPrefetchAsync(um_x, sizeof(float) * num_rows, cudaCpuDeviceId));
    CUDA_RT_CALL(cudaMemPrefetchAsync(dot_result, sizeof(double), cudaCpuDeviceId));

    for (int gpu_idx = 0; gpu_idx < num_devices; gpu_idx++) {
        CUDA_RT_CALL(cudaSetDevice(gpu_idx));
        CUDA_RT_CALL(cudaStreamSynchronize(nStreams[gpu_idx]));
    }

    r1 = (float)*dot_result;

    double stop = omp_get_wtime();

    for (int gpu_idx = 0; gpu_idx < num_devices; gpu_idx++) {
        CUDA_RT_CALL(cudaSetDevice(gpu_idx));

        if (compare_to_single_gpu) {
            for (int i = 0; i < num_rows; i++) {
                x_host[i] = um_x[i];
            }
        }

        if (gpu_idx == 0) {
            report_results(num_rows, x_ref_host, x_host, num_devices, single_gpu_runtime, start,
                           stop, compare_to_single_gpu);

            CUDA_RT_CALL(cudaFreeHost(x_host));
            CUDA_RT_CALL(cudaFreeHost(x_ref_host));
        }
    }

    CUDA_RT_CALL(cudaFreeHost(multi_device_data.hostMemoryArrivedList));
    CUDA_RT_CALL(cudaFree(um_I));
    CUDA_RT_CALL(cudaFree(um_J));
    CUDA_RT_CALL(cudaFree(um_val));
    CUDA_RT_CALL(cudaFree(um_x));
    CUDA_RT_CALL(cudaFree(um_r));
    CUDA_RT_CALL(cudaFree(um_p));
    CUDA_RT_CALL(cudaFree(um_s));
    CUDA_RT_CALL(cudaFree(dot_result));
    free(host_val);

    return 0;
}