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

#include "../../include/baseline/discrete-non-pipelined.cuh"
#include "../../include/common.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace BaselineDiscreteNonPipelined {

__device__ double grid_dot_result = 0.0;

__global__ void gpuDotProduct(real *vecA, real *vecB, int num_rows, const int device_rank,
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

__global__ void addLocalDotContribution(double *dot_result) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        atomicAdd_system(dot_result, grid_dot_result);
        grid_dot_result = 0.0;
    }
}

__global__ void resetLocalDotProduct(double *dot_result) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        *dot_result = 0.0;
    }
}

}  // namespace BaselineDiscreteNonPipelined

int BaselineDiscreteNonPipelined::init(int argc, char *argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 10000);
    std::string matrix_path_str = get_argval<std::string>(argv, argv + argc, "-matrix_path", "");
    const bool compare_to_single_gpu = get_arg(argv, argv + argc, "-compare-single-gpu");
    const bool compare_to_cpu = get_arg(argv, argv + argc, "-compare-cpu");

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
    real *host_val = NULL;
    real *x_ref_host = NULL;

    real *x_ref_single_gpu = NULL;

    real *s_cpu = NULL;
    real *r_cpu = NULL;
    real *p_cpu = NULL;
    real *x_ref_cpu = NULL;

    int *um_I = NULL;
    int *um_J = NULL;
    real *um_val = NULL;

    real *um_x;
    real *um_r;
    real *um_p;
    real *um_s;
    real *um_ax0;

    double *um_tmp_dot_delta1;
    double *um_tmp_dot_gamma1;
    real *um_tmp_dot_delta0;
    real *um_tmp_dot_gamma0;

    real *um_alpha;
    real *um_negative_alpha;
    real *um_beta;

    real real_positive_one = 1.0;
    real real_negative_one = -1.0;

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
        CUDA_RT_CALL(cudaMallocManaged((void **)&um_val, sizeof(real) * nnz));

        host_val = (real *)malloc(sizeof(real) * nnz);

        /* Generate a random tridiagonal symmetric matrix in CSR format */
        genTridiag(um_I, um_J, host_val, num_rows, nnz);

        memcpy(um_val, host_val, sizeof(real) * nnz);

    } else {
        if (loadMMSparseMatrix<real>(matrix_path_char, 'd', true, &num_rows, &num_cols, &nnz,
                                     &host_val, &host_I, &host_J, true)) {
            exit(EXIT_FAILURE);
        }

        CUDA_RT_CALL(cudaMallocManaged((void **)&um_I, sizeof(int) * (num_rows + 1)));
        CUDA_RT_CALL(cudaMallocManaged((void **)&um_J, sizeof(int) * nnz));
        CUDA_RT_CALL(cudaMallocManaged((void **)&um_val, sizeof(real) * nnz));

        memcpy(um_I, host_I, sizeof(int) * (num_rows + 1));
        memcpy(um_J, host_J, sizeof(int) * nnz);
        memcpy(um_val, host_val, sizeof(real) * nnz);
    }

    CUDA_RT_CALL(cudaMallocManaged((void **)&um_x, sizeof(real) * num_rows));

    // Comparing to Single GPU Non-Persistent Non-Pipelined implementation
#pragma omp parallel num_threads(num_devices)
    {
        int gpu_idx = omp_get_thread_num();

        if (gpu_idx == 0) {
            if (compare_to_single_gpu) {
                CUDA_RT_CALL(cudaSetDevice(gpu_idx));

                CUDA_RT_CALL(cudaMallocHost(&x_ref_single_gpu, num_rows * sizeof(real)));

                single_gpu_runtime = SingleGPUStandardDiscrete::run_single_gpu(
                    iter_max, um_I, um_J, um_val, x_ref_single_gpu, num_rows, nnz);

                // single_gpu_runtime = SingleGPUPipelinedDiscrete::run_single_gpu(
                //     iter_max, um_I, um_J, um_val, x_ref_single_gpu, num_rows, nnz);
            }

            if (compare_to_cpu) {
                s_cpu = (real *)malloc(sizeof(real) * num_rows);
                r_cpu = (real *)malloc(sizeof(real) * num_rows);
                p_cpu = (real *)malloc(sizeof(real) * num_rows);
                x_ref_cpu = (real *)malloc(sizeof(real) * num_rows);

                for (int i = 0; i < num_rows; i++) {
                    r_cpu[i] = 1.0;
                    s_cpu[i] = 0.0;
                    x_ref_cpu[i] = 0.0;
                }

                CPU::cpuConjugateGrad(iter_max, host_I, host_J, host_val, x_ref_cpu, s_cpu, p_cpu,
                                      r_cpu, nnz, num_rows, tol);
            }
        }
    }

    double *um_dot_result;
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_dot_result, sizeof(double)));

    CUDA_RT_CALL(cudaMemset(um_dot_result, 0, sizeof(double)));

    CUDA_RT_CALL(cudaMallocManaged((void **)&um_x, sizeof(real) * num_rows));

    CUDA_RT_CALL(cudaMallocManaged((void **)&um_tmp_dot_delta1, sizeof(double)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_tmp_dot_gamma1, sizeof(double)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_tmp_dot_delta0, sizeof(real)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_tmp_dot_gamma0, sizeof(real)));

    CUDA_RT_CALL(cudaMemset(um_tmp_dot_delta1, 0, sizeof(double)));
    CUDA_RT_CALL(cudaMemset(um_tmp_dot_gamma1, 0, sizeof(double)));
    CUDA_RT_CALL(cudaMemset(um_tmp_dot_delta0, 0, sizeof(real)));
    CUDA_RT_CALL(cudaMemset(um_tmp_dot_gamma0, 0, sizeof(real)));

    // temp memory for ConjugateGradient
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_r, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_p, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_s, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_ax0, num_rows * sizeof(real)));

    CUDA_RT_CALL(cudaMallocManaged((void **)&um_alpha, sizeof(real)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_negative_alpha, sizeof(real)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_beta, sizeof(real)));
    // ASSUMPTION: All GPUs are the same and P2P callable

    cudaStream_t nStreams[num_devices];

    int sMemSize = sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1);

    CUDA_RT_CALL(cudaSetDevice(0));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int numSms = deviceProp.multiProcessorCount;

    // Structure used for cross-grid synchronization.
    unsigned char *hostMemoryArrivedList;

    CUDA_RT_CALL(cudaHostAlloc((void **)&hostMemoryArrivedList,
                               (num_devices - 1) * sizeof(*hostMemoryArrivedList),
                               cudaHostAllocPortable));
    memset(hostMemoryArrivedList, 0, (num_devices - 1) * sizeof(*hostMemoryArrivedList));

    int numBlocksInitVectorsPerSM = 0;
    int numBlocksSpmvPerSM = 0;
    int numBlocksSaxpyPerSM = 0;
    int numBlocksDotProductPerSM = 0;
    int numBlocksCopyVectorPerSM = 0;
    int numBlocksScaleVectorAndSaxpyPerSM = 0;

    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksInitVectorsPerSM, MultiGPU::initVectors, THREADS_PER_BLOCK, 0));
    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksSpmvPerSM, MultiGPU::gpuSpMV, THREADS_PER_BLOCK, 0));
    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksSaxpyPerSM, MultiGPU::gpuSaxpy, THREADS_PER_BLOCK, 0));
    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksDotProductPerSM, gpuDotProduct, THREADS_PER_BLOCK, sMemSize));
    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksCopyVectorPerSM, MultiGPU::gpuCopyVector, THREADS_PER_BLOCK, 0));
    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksScaleVectorAndSaxpyPerSM,
                                                               MultiGPU::gpuScaleVectorAndSaxpy,
                                                               THREADS_PER_BLOCK, 0));

    int initVectorsGridSize = numBlocksInitVectorsPerSM * numSms;
    int spmvGridSize = numBlocksSpmvPerSM * numSms;
    int saxpyGridSize = numBlocksSaxpyPerSM * numSms;
    int dotProductGridSize = numBlocksDotProductPerSM * numSms;
    int copyVectorGridSize = numBlocksCopyVectorPerSM * numSms;
    int scaleVectorAndSaxpyGridSize = numBlocksScaleVectorAndSaxpyPerSM * numSms;

    double start = omp_get_wtime();

#pragma omp parallel num_threads(num_devices)
    {
        int gpu_idx = omp_get_thread_num();

        CUDA_RT_CALL(cudaSetDevice(gpu_idx));
        CUDA_RT_CALL(cudaStreamCreate(&nStreams[gpu_idx]));

        MultiGPU::initVectors<<<initVectorsGridSize, THREADS_PER_BLOCK, 0, nStreams[gpu_idx]>>>(
            um_r, um_x, num_rows, gpu_idx, num_devices);

        MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                            hostMemoryArrivedList);

        CUDA_RT_CALL(cudaDeviceSynchronize());

        // ax0 = Ax0
        MultiGPU::gpuSpMV<<<spmvGridSize, THREADS_PER_BLOCK, 0, nStreams[gpu_idx]>>>(
            um_I, um_J, um_val, nnz, num_rows, real_positive_one, um_x, um_ax0, gpu_idx,
            num_devices);

        MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                            hostMemoryArrivedList);

        CUDA_RT_CALL(cudaDeviceSynchronize());

        // r0 = b0 - ax0
        // NOTE: b is a unit vector.
        MultiGPU::gpuSaxpy<<<saxpyGridSize, THREADS_PER_BLOCK, 0, nStreams[gpu_idx]>>>(
            um_ax0, um_r, real_negative_one, num_rows, gpu_idx, num_devices);

        MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                            hostMemoryArrivedList);

        CUDA_RT_CALL(cudaDeviceSynchronize());

        // p0 = r0
        MultiGPU::gpuCopyVector<<<copyVectorGridSize, THREADS_PER_BLOCK, 0, nStreams[gpu_idx]>>>(
            um_r, um_p, num_rows, gpu_idx, num_devices);

        MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                            hostMemoryArrivedList);

        CUDA_RT_CALL(cudaDeviceSynchronize());

        resetLocalDotProduct<<<1, 1, 0, nStreams[gpu_idx]>>>(um_tmp_dot_gamma1);

        MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                            hostMemoryArrivedList);

        CUDA_RT_CALL(cudaDeviceSynchronize());

        gpuDotProduct<<<dotProductGridSize, THREADS_PER_BLOCK, sMemSize, nStreams[gpu_idx]>>>(
            um_r, um_r, num_rows, gpu_idx, num_devices);

        MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                            hostMemoryArrivedList);

        CUDA_RT_CALL(cudaDeviceSynchronize());

        addLocalDotContribution<<<1, 1, 0, nStreams[gpu_idx]>>>(um_tmp_dot_gamma1);

        MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                            hostMemoryArrivedList);

        CUDA_RT_CALL(cudaDeviceSynchronize());

        *um_tmp_dot_gamma0 = (real)*um_tmp_dot_gamma1;

        MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                            hostMemoryArrivedList);

        CUDA_RT_CALL(cudaDeviceSynchronize());

        int k = 1;

        while (k <= iter_max) {
            // SpMV
            MultiGPU::gpuSpMV<<<spmvGridSize, THREADS_PER_BLOCK, 0, nStreams[gpu_idx]>>>(
                um_I, um_J, um_val, nnz, num_rows, real_positive_one, um_p, um_s, gpu_idx,
                num_devices);

            MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                                hostMemoryArrivedList);

            CUDA_RT_CALL(cudaDeviceSynchronize());

            resetLocalDotProduct<<<1, 1, 0, nStreams[gpu_idx]>>>(um_tmp_dot_delta1);

            MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                                hostMemoryArrivedList);

            CUDA_RT_CALL(cudaDeviceSynchronize());

            gpuDotProduct<<<dotProductGridSize, THREADS_PER_BLOCK, sMemSize, nStreams[gpu_idx]>>>(
                um_p, um_s, num_rows, gpu_idx, num_devices);

            MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                                hostMemoryArrivedList);

            CUDA_RT_CALL(cudaDeviceSynchronize());

            addLocalDotContribution<<<1, 1, 0, nStreams[gpu_idx]>>>(um_tmp_dot_delta1);

            MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                                hostMemoryArrivedList);

            CUDA_RT_CALL(cudaDeviceSynchronize());

            r1_div_x<<<1, 1, 0, nStreams[gpu_idx]>>>(*um_tmp_dot_gamma0, (real)*um_tmp_dot_delta1,
                                                     um_alpha);

            MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                                hostMemoryArrivedList);

            CUDA_RT_CALL(cudaDeviceSynchronize());

            // x_(k+1) = x_k + alpha_k * p_k
            MultiGPU::gpuSaxpy<<<saxpyGridSize, THREADS_PER_BLOCK, 0, nStreams[gpu_idx]>>>(
                um_p, um_x, *um_alpha, num_rows, gpu_idx, num_devices);

            MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                                hostMemoryArrivedList);

            CUDA_RT_CALL(cudaDeviceSynchronize());

            a_minus<<<1, 1, 0, nStreams[gpu_idx]>>>(*um_alpha, um_negative_alpha);

            MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                                hostMemoryArrivedList);

            CUDA_RT_CALL(cudaDeviceSynchronize());

            // r_(k+1) = r_k - alpha_k * s
            MultiGPU::gpuSaxpy<<<saxpyGridSize, THREADS_PER_BLOCK, 0, nStreams[gpu_idx]>>>(
                um_s, um_r, *um_negative_alpha, num_rows, gpu_idx, num_devices);

            MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                                hostMemoryArrivedList);

            CUDA_RT_CALL(cudaDeviceSynchronize());

            resetLocalDotProduct<<<1, 1, 0, nStreams[gpu_idx]>>>(um_tmp_dot_gamma1);

            MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                                hostMemoryArrivedList);

            CUDA_RT_CALL(cudaDeviceSynchronize());

            gpuDotProduct<<<dotProductGridSize, THREADS_PER_BLOCK, sMemSize, nStreams[gpu_idx]>>>(
                um_r, um_r, num_rows, gpu_idx, num_devices);

            MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                                hostMemoryArrivedList);

            CUDA_RT_CALL(cudaDeviceSynchronize());

            addLocalDotContribution<<<1, 1, 0, nStreams[gpu_idx]>>>(um_tmp_dot_gamma1);

            MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                                hostMemoryArrivedList);

            CUDA_RT_CALL(cudaDeviceSynchronize());

            r1_div_x<<<1, 1, 0, nStreams[gpu_idx]>>>((real)*um_tmp_dot_gamma1, *um_tmp_dot_gamma0,
                                                     um_beta);

            MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                                hostMemoryArrivedList);

            CUDA_RT_CALL(cudaDeviceSynchronize());

            // p_(k+1) = r_(k+1) = beta_k * p_(k)
            MultiGPU::gpuScaleVectorAndSaxpy<<<scaleVectorAndSaxpyGridSize, THREADS_PER_BLOCK, 0,
                                               nStreams[gpu_idx]>>>(
                um_r, um_p, real_positive_one, *um_beta, num_rows, gpu_idx, num_devices);

            MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                                hostMemoryArrivedList);

            CUDA_RT_CALL(cudaDeviceSynchronize());

            *um_tmp_dot_delta0 = (real)*um_tmp_dot_delta1;
            *um_tmp_dot_gamma0 = (real)*um_tmp_dot_gamma1;

            MultiGPU::syncPeers<<<1, 1, 0, nStreams[gpu_idx]>>>(gpu_idx, num_devices,
                                                                hostMemoryArrivedList);

            CUDA_RT_CALL(cudaDeviceSynchronize());

#pragma omp barrier

            k++;
        }
    }

    // r1 = (real)um_dot_result[0];

    double stop = omp_get_wtime();

    for (int gpu_idx = 0; gpu_idx < num_devices; gpu_idx++) {
        CUDA_RT_CALL(cudaSetDevice(gpu_idx));

        if (gpu_idx == 0) {
            report_results(num_rows, x_ref_single_gpu, x_ref_cpu, um_x, num_devices,
                           single_gpu_runtime, start, stop, compare_to_single_gpu, compare_to_cpu);

            CUDA_RT_CALL(cudaFreeHost(x_ref_single_gpu));
        }
    }

    CUDA_RT_CALL(cudaFreeHost(hostMemoryArrivedList));
    CUDA_RT_CALL(cudaFree(um_I));
    CUDA_RT_CALL(cudaFree(um_J));
    CUDA_RT_CALL(cudaFree(um_val));
    CUDA_RT_CALL(cudaFree(um_x));
    CUDA_RT_CALL(cudaFree(um_r));
    CUDA_RT_CALL(cudaFree(um_p));
    CUDA_RT_CALL(cudaFree(um_ax0));
    CUDA_RT_CALL(cudaFree(um_dot_result));
    free(host_val);

    return 0;
}
