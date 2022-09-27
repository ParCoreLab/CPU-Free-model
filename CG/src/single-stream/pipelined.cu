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

#include "../../include/common.h"
#include "../../include/single-stream/pipelined.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace SingleStreamPipelined {

__device__ double grid_dot_result_delta = 0.0;
__device__ double grid_dot_result_gamma = 0.0;

__device__ void gpuSpMV(int *I, int *J, float *val, int nnz, int num_rows, float alpha,
                        float *inputVecX, float *outputVecY, const int num_allocated_tbs,
                        const int device_rank, const int num_devices, const PeerGroup &peer_group) {
    int subgrid_thread_rank = peer_group.calc_subgrid_thread_rank(num_allocated_tbs);
    int subgrid_size = peer_group.calc_subgrid_size(num_allocated_tbs);

    for (int i = subgrid_thread_rank; i < num_rows; i += subgrid_size) {
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

// Performs two dot products at the same time
// Used to perform <r, r> and <r, w> at the same time
// Can we combined the two atomicAdds somehow?
__device__ void gpuDotProductsMerged(float *vecA_delta, float *vecB_delta, float *vecA_gamma,
                                     float *vecB_gamma, int num_rows, const cg::thread_block &cta,
                                     const int num_allocated_tbs, const int device_rank,
                                     const int num_devices, const int sMemSize,
                                     const PeerGroup &peer_group) {
    // First half (up to sMemSize / 2) will be used for delta
    // Second half (from sMemSize / 2) will be used for gamma
    extern __shared__ double tmp[];

    double *tmp_delta = (double *)tmp;
    double *tmp_gamma = (double *)&tmp_delta[sMemSize / (2 * sizeof(double))];

    double temp_sum_delta = 0.0;
    double temp_sum_gamma = 0.0;

    int subgrid_thread_rank = peer_group.calc_subgrid_thread_rank(num_allocated_tbs);
    int subgrid_size = peer_group.calc_subgrid_size(num_allocated_tbs);

    for (int i = subgrid_thread_rank; i < num_rows; i += subgrid_size) {
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
            atomicAdd(&grid_dot_result_delta, temp_sum_delta);
            atomicAdd(&grid_dot_result_gamma, temp_sum_gamma);
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

__global__ void multiGpuConjugateGradient(int *I, int *J, float *val, float *x, float *s, float *p,
                                          float *r, float *w, float *t, float *u,
                                          double *dot_result_delta, double *dot_result_gamma,
                                          int nnz, int N, float tol,
                                          MultiDeviceData multi_device_data, const int iter_max,
                                          const int num_blocks_for_spmv, const int sMemSize) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    PeerGroup peer_group(multi_device_data, grid);

    const int num_devices = multi_device_data.numDevices;
    const int device_rank = multi_device_data.deviceRank;
    const int num_blocks_for_dot = gridDim.x - num_blocks_for_spmv;

    float float_positive_one = 1.0;
    float float_negative_one = -1.0;

    float tmp_dot_delta_0 = 0.0;
    float tmp_dot_gamma_0 = 0.0;

    float tmp_dot_delta_1;
    float tmp_dot_gamma_1;

    float b;
    float a;
    float na;

    for (int i = peer_group.thread_rank(); i < N; i += peer_group.size()) {
        r[i] = 1.0;
        x[i] = 0.0;
    }

    cg::sync(grid);

    gpuSpMV(I, J, val, nnz, N, float_positive_one, x, s, gridDim.x, device_rank, num_devices,
            peer_group);

    cg::sync(grid);

    gpuSaxpy(s, r, float_negative_one, N, peer_group);

    cg::sync(grid);

    gpuSpMV(I, J, val, nnz, N, float_positive_one, r, w, gridDim.x, device_rank, num_devices,
            peer_group);

    cg::sync(grid);

    gpuDotProductsMerged(r, r, r, w, N, cta, gridDim.x, device_rank, num_devices, sMemSize,
                         peer_group);

    cg::sync(grid);

    if (grid.thread_rank() == 0) {
        atomicAdd_system(dot_result_delta, grid_dot_result_delta);
        atomicAdd_system(dot_result_gamma, grid_dot_result_gamma);

        grid_dot_result_delta = 0.0;
        grid_dot_result_gamma = 0.0;
    }

    peer_group.sync();

    // I don't know if this is correct.
    // Need to figure the whole tolerance thing for pipelined CG
    tmp_dot_delta_1 = *dot_result_delta;

    int k = 1;

    while (k <= iter_max) {
        if (k > 1) {
            b = tmp_dot_delta_1 / tmp_dot_delta_0;

            // gpuScaleVectorAndSaxpy(r, p, float_positive_one, b, N, peer_group);
            // gpuScaleVectorAndSaxpy(w, s, float_positive_one, b, N, peer_group);
            // gpuScaleVectorAndSaxpy(t, u, float_positive_one, b, N, peer_group);

        } else {
            gpuCopyVector(r, p, N, peer_group);

            // Need to figure out what to copy where
            // Other vectors also need to be initialized
            // Fine for now
        }

        peer_group.sync();

        if (peer_group.thread_rank() == 0) {
            *dot_result_delta = 0.0;
            *dot_result_gamma = 0.0;
        }

        peer_group.sync();

        // This is where the overlap will happen
        // Need to figure out how to do it
        // Need to keep track of peer_group.sync() calls

        // if => SpMV
        // else => Dot

        if (blockIdx.x < num_blocks_for_spmv) {
            gpuSpMV(I, J, val, nnz, N, float_positive_one, p, s, num_blocks_for_spmv, device_rank,
                    num_devices, peer_group);
        } else {
            cg::coalesced_group dot_active_threads = cg::coalesced_threads();

            gpuDotProductsMerged(r, r, r, w, N, cta, num_blocks_for_dot, device_rank, num_devices,
                                 sMemSize, peer_group);

            cg::sync(dot_active_threads);

            if (dot_active_threads.thread_rank() == 0) {
                atomicAdd_system(dot_result_delta, grid_dot_result_delta);
                atomicAdd_system(dot_result_gamma, grid_dot_result_gamma);

                grid_dot_result_delta = 0.0;
                grid_dot_result_gamma = 0.0;
            }
        }

        peer_group.sync();

        a = tmp_dot_delta_1 / (tmp_dot_gamma_1 - (b / a) * tmp_dot_delta_1);

        gpuSaxpy(p, x, a, N, peer_group);

        na = -a;

        gpuSaxpy(s, r, na, N, peer_group);

        // Do we need this sync?
        peer_group.sync();

        tmp_dot_delta_0 = tmp_dot_delta_1;
        tmp_dot_gamma_0 = tmp_dot_gamma_1;

        tmp_dot_delta_1 = *dot_result_delta;
        tmp_dot_gamma_1 = *dot_result_gamma;

        k++;
    }
}
}  // namespace SingleStreamPipelined

int SingleStreamPipelined::init(int argc, char *argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 10000);
    std::string matrix_path_str = get_argval<std::string>(argv, argv + argc, "-matrix_path", "");
    int num_blocks_for_spmv = get_argval<int>(argv, argv + argc, "-num-spmv-blocks", -1);
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

    float *um_r;
    float *um_p;
    float *um_s;
    float *um_x;
    float *um_w;
    float *um_u;
    float *um_t;

    float r1;

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

// Comparing to Single GPU Non-Persistent Non-Pipelined implementation
#pragma omp parallel num_threads(num_devices)
    {
        int gpu_idx = omp_get_thread_num();

        if (compare_to_single_gpu && gpu_idx == 0) {
            CUDA_RT_CALL(cudaSetDevice(gpu_idx));

            CUDA_RT_CALL(cudaMallocHost(&x_ref_host, num_rows * sizeof(float)));
            CUDA_RT_CALL(cudaMallocHost(&x_host, num_rows * sizeof(float)));

            single_gpu_runtime = SingleGPUPipelinedNonPersistent::run_single_gpu(
                iter_max, matrix_path_char, generate_random_tridiag_matrix, um_I, um_J, um_val,
                x_ref_host, num_rows, nnz);
        }
    }

    CUDA_RT_CALL(cudaMallocManaged((void **)&um_x, sizeof(float) * num_rows));

    double *um_dot_result_delta;
    double *um_dot_result_gamma;

    CUDA_RT_CALL(cudaMallocManaged((void **)&um_dot_result_delta, sizeof(double)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_dot_result_gamma, sizeof(double)));

    CUDA_RT_CALL(cudaMemset(um_dot_result_delta, 0, sizeof(double)));
    CUDA_RT_CALL(cudaMemset(um_dot_result_gamma, 0, sizeof(double)));

    // temp memory for ConjugateGradient
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_r, num_rows * sizeof(float)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_p, num_rows * sizeof(float)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_s, num_rows * sizeof(float)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_w, num_rows * sizeof(float)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_u, num_rows * sizeof(float)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_t, num_rows * sizeof(float)));

    // ASSUMPTION: All GPUs are the same and P2P callable

    int sMemSize = 2 * (sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1));
    int numThreads = THREADS_PER_BLOCK;

    CUDA_RT_CALL(cudaSetDevice(0));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int numSms = deviceProp.multiProcessorCount;

    int numBlocksPerSm = INT_MAX;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, multiGpuConjugateGradient,
                                                  numThreads, sMemSize);

    dim3 dimGrid(numSms * numBlocksPerSm, 1, 1), dimBlock(THREADS_PER_BLOCK, 1, 1);

    // TODO: Use rough heuristic to know SpMV vs Dot TB allocation
    // Look at paper that compare breakdown of components
    // Can also determine experimentally
    // Hardcoding for now => half and half to each

    // If number of blocks for SpMV is not specified, split evenly between SpMV and Dot
    if (num_blocks_for_spmv == -1) {
        std::cout << "Splitting thread blocks evenly between SpMV and Dot" << std::endl;
        num_blocks_for_spmv = (numSms * numBlocksPerSm) / 2;
    }

    // Structure used for cross-grid synchronization.
    unsigned char *hostMemoryArrivedList;

    CUDA_RT_CALL(cudaHostAlloc((void **)&hostMemoryArrivedList,
                               (num_devices - 1) * sizeof(*hostMemoryArrivedList),
                               cudaHostAllocPortable));
    memset(hostMemoryArrivedList, 0, (num_devices - 1) * sizeof(*hostMemoryArrivedList));

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
        (void *)&um_x,
        (void *)&um_s,
        (void *)&um_p,
        (void *)&um_r,
        (void *)&um_w,
        (void *)&um_t,
        (void *)&um_u,
        (void *)&um_dot_result_delta,
        (void *)&um_dot_result_gamma,
        (void *)&nnz,
        (void *)&num_rows,
        (void *)&tol,
        (void *)&multi_device_data,
        (void *)&iter_max,
        (void *)&num_blocks_for_spmv,
        (void *)&sMemSize,
    };

    cudaStream_t nStreams[num_devices];

    for (int gpu_idx = 0; gpu_idx < num_devices; gpu_idx++) {
        CUDA_RT_CALL(cudaSetDevice(gpu_idx));
        CUDA_RT_CALL(cudaStreamCreate(&nStreams[gpu_idx]));
    }

    double start = omp_get_wtime();

    for (int gpu_idx = 0; gpu_idx < num_devices; gpu_idx++) {
        CUDA_RT_CALL(cudaSetDevice(gpu_idx));
        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *)multiGpuConjugateGradient, dimGrid,
                                                 dimBlock, kernelArgs, sMemSize,
                                                 nStreams[gpu_idx]));

        multi_device_data.deviceRank++;
    }

    for (int gpu_idx = 0; gpu_idx < num_devices; gpu_idx++) {
        CUDA_RT_CALL(cudaSetDevice(gpu_idx));
        CUDA_RT_CALL(cudaStreamSynchronize(nStreams[gpu_idx]));
    }

    r1 = (float)um_dot_result_gamma[0];

    double stop = omp_get_wtime();

    for (int i = 0; i < num_rows; i++) {
        x_host[i] = um_x[i];
    }

    for (int gpu_idx = 0; gpu_idx < num_devices; gpu_idx++) {
        CUDA_RT_CALL(cudaSetDevice(gpu_idx));

        if (compare_to_single_gpu && gpu_idx == 0) {
            report_results(num_rows, x_ref_host, x_host, num_devices, single_gpu_runtime, start,
                           stop, compare_to_single_gpu);
        }
    }

    CUDA_RT_CALL(cudaFreeHost(hostMemoryArrivedList));
    CUDA_RT_CALL(cudaFree(um_I));
    CUDA_RT_CALL(cudaFree(um_J));
    CUDA_RT_CALL(cudaFree(um_val));
    CUDA_RT_CALL(cudaFree(um_x));
    CUDA_RT_CALL(cudaFree(um_r));
    CUDA_RT_CALL(cudaFree(um_p));
    CUDA_RT_CALL(cudaFree(um_s));
    CUDA_RT_CALL(cudaFree(um_dot_result_delta));
    CUDA_RT_CALL(cudaFree(um_dot_result_gamma));

    free(host_val);

    return 0;
}
