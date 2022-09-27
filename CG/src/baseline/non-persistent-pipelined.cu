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

#include "../../include/baseline/non-persistent-pipelined.cuh"
#include "../../include/common.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace BaselineNonPersistentPipelined {

#define ENABLE_CPU_DEBUG_CODE 0

// delta => <r, r>
// gamma => <r, w>
__device__ double grid_dot_result_delta = 0.0;
__device__ double grid_dot_result_gamma = 0.0;

__global__ void initVectors(float *r, float *x, int num_rows, const int device_rank,
                            const int num_devices) {
    size_t local_grid_size = gridDim.x * blockDim.x;
    size_t local_grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    size_t global_grid_size = local_grid_size * num_devices;
    size_t global_grid_rank = device_rank * local_grid_size + local_grid_rank;

    for (size_t i = global_grid_rank; i < num_rows; i += global_grid_size) {
        r[i] = 1.0;
        x[i] = 0.0;
    }
}

__global__ void update_a_k(float *dot_delta_1, float *dot_gamma_1, float *b, float *a) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        a[0] = dot_delta_1[0] / (dot_gamma_1[0] - (b[0] / a[0]) * dot_delta_1[0]);
    }
}

__global__ void update_b_k(float *dot_delta_1, float *dot_delta_0, float *b) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        b[0] = dot_delta_1[0] / dot_delta_0[0];
    }
}

__global__ void r1_div_x(float *r1, float *r0, float *b) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        b[0] = r1[0] / r0[0];
    }
}

__global__ void a_minus(float *a, float *na) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        na[0] = -(a[0]);
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

__global__ void gpuSaxpy(float *x, float *y, float *a, int num_rows, const int device_rank,
                         const int num_devices) {
    size_t local_grid_size = gridDim.x * blockDim.x;
    size_t local_grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    size_t global_grid_size = local_grid_size * num_devices;
    size_t global_grid_rank = device_rank * local_grid_size + local_grid_rank;

    for (size_t i = global_grid_rank; i < num_rows; i += global_grid_size) {
        y[i] = a[0] * x[i] + y[i];
    }
}

// Performs two dot products at the same time
// Used to perform <r, r> and <r, w> at the same time
// Can we combined the two atomicAdds somehow?

__global__ void gpuDotProductsMerged(float *vecA_delta, float *vecB_delta, float *vecA_gamma,
                                     float *vecB_gamma, int num_rows, const int device_rank,
                                     const int num_devices, const int sMemSize) {
    cg::thread_block cta = cg::this_thread_block();

    size_t local_grid_size = gridDim.x * blockDim.x;
    size_t local_grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    size_t global_grid_size = local_grid_size * num_devices;
    size_t global_grid_rank = device_rank * local_grid_size + local_grid_rank;

    extern __shared__ double tmp[];

    double *tmp_delta = (double *)tmp;
    double *tmp_gamma = (double *)&tmp_delta[sMemSize / (2 * sizeof(double))];

    double temp_sum_delta = 0.0;
    double temp_sum_gamma = 0.0;

    for (size_t i = global_grid_rank; i < num_rows; i += global_grid_size) {
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

__global__ void gpuCopyVector(float *srcA, float *destB, int num_rows, const int device_rank,
                              const int num_devices) {
    size_t local_grid_size = gridDim.x * blockDim.x;
    size_t local_grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    size_t global_grid_size = local_grid_size * num_devices;
    size_t global_grid_rank = device_rank * local_grid_size + local_grid_rank;

    for (size_t i = global_grid_rank; i < num_rows; i += global_grid_size) {
        destB[i] = srcA[i];
    }
}

__global__ void gpuScaleVectorAndSaxpy(float *x, float *y, float a, float *scale, int num_rows,
                                       const int device_rank, const int num_devices) {
    size_t local_grid_size = gridDim.x * blockDim.x;
    size_t local_grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    size_t global_grid_size = local_grid_size * num_devices;
    size_t global_grid_rank = device_rank * local_grid_size + local_grid_rank;

    for (size_t i = global_grid_rank; i < num_rows; i += global_grid_size) {
        y[i] = a * x[i] + scale[0] * y[i];
    }
}

__global__ void addLocalDotContributions(double *dot_result_delta, double *dot_result_gamma) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        atomicAdd_system(dot_result_delta, grid_dot_result_delta);
        atomicAdd_system(dot_result_gamma, grid_dot_result_gamma);

        grid_dot_result_delta = 0.0;
        grid_dot_result_gamma = 0.0;
    }
}

__device__ unsigned char load_arrived(unsigned char *arrived) {
#if __CUDA_ARCH__ < 700
    return *(volatile unsigned char *)arrived;
#else
    unsigned int result;
    asm volatile("ld.acquire.sys.global.u8 %0, [%1];" : "=r"(result) : "l"(arrived) : "memory");
    return result;
#endif
}

__device__ void store_arrived(unsigned char *arrived, unsigned char val) {
#if __CUDA_ARCH__ < 700
    *(volatile unsigned char *)arrived = val;
#else
    unsigned int reg_val = val;
    asm volatile("st.release.sys.global.u8 [%1], %0;" ::"r"(reg_val) "l"(arrived) : "memory");

    // Avoids compiler warnings from unused variable val.
    (void)(reg_val = reg_val);
#endif
}

__global__ void syncPeers(const int device_rank, const int num_devices,
                          unsigned char *hostMemoryArrivedList) {
    int local_grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    // One thread from each grid participates in the sync.
    if (local_grid_rank == 0) {
        if (device_rank == 0) {
            // Leader grid waits for others to join and then releases them.
            // Other GPUs can arrive in any order, so the leader have to wait for
            // all others.

            for (int i = 0; i < num_devices - 1; i++) {
                while (load_arrived(&hostMemoryArrivedList[i]) == 0)
                    ;
            }

            for (int i = 0; i < num_devices - 1; i++) {
                store_arrived(&hostMemoryArrivedList[i], 0);
            }

            __threadfence_system();
        } else {
            // Other grids note their arrival and wait to be released.
            store_arrived(&hostMemoryArrivedList[device_rank - 1], 1);

            while (load_arrived(&hostMemoryArrivedList[device_rank - 1]) == 1)
                ;
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
}  // namespace BaselineNonPersistentPipelined

int BaselineNonPersistentPipelined::init(int argc, char *argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 10000);
    std::string matrix_path_str = get_argval<std::string>(argv, argv + argc, "-matrix_path", "");
    const bool compare_to_cpu = get_arg(argv, argv + argc, "-compare");
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

    float *um_tmp_dot_delta1;
    float *um_tmp_dot_delta0;
    float *um_tmp_dot_gamma1;
    float *um_tmp_dot_gamma0;
    float *um_a;
    float *um_na;
    float *um_b;
    float *um_float_negative_one;

    const float tol = 1e-5f;
    float rhs = 1.0;
    float r1;
    float float_positive_one = 1.0;
    float float_negative_one = -1.0;

    cudaStream_t streamsOtherOps[num_devices];
    cudaStream_t streamsSaxpy[num_devices];
    cudaStream_t streamsDot[num_devices];
    cudaStream_t streamsSpMV[num_devices];

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

    CUDA_RT_CALL(cudaMallocManaged((void **)&um_x, sizeof(float) * num_rows));

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

    CUDA_RT_CALL(cudaMallocManaged((void **)&um_tmp_dot_delta1, sizeof(float)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_tmp_dot_delta0, sizeof(float)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_tmp_dot_gamma1, sizeof(float)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_tmp_dot_gamma0, sizeof(float)));

    CUDA_RT_CALL(cudaMallocManaged((void **)&um_a, sizeof(float)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_na, sizeof(float)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_b, sizeof(float)));

    CUDA_RT_CALL(cudaMalloc((void **)&um_float_negative_one, sizeof(float)));
    CUDA_RT_CALL(cudaMemcpy(um_float_negative_one, &float_negative_one, sizeof(float),
                            cudaMemcpyHostToDevice));

    // ASSUMPTION: All GPUs are the same and P2P callable

    // Multiplying by 2 because the two dot products are merged
    int sMemSize = 2 * (sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1));

    CUDA_RT_CALL(cudaSetDevice(0));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int numSms = deviceProp.multiProcessorCount;

#if ENABLE_CPU_DEBUG_CODE
    float *s_cpu = (float *)malloc(sizeof(float) * N);
    float *r_cpu = (float *)malloc(sizeof(float) * N);
    float *p_cpu = (float *)malloc(sizeof(float) * N);
    float *x_cpu = (float *)malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        r_cpu[i] = 1.0;
        s_cpu[i] = x_cpu[i] = 0.0;
    }
#endif

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

    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksInitVectorsPerSM, gpuSpMV,
                                                               THREADS_PER_BLOCK, 0));
    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksSpmvPerSM, gpuSpMV,
                                                               THREADS_PER_BLOCK, 0));
    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksSaxpyPerSM, gpuSaxpy,
                                                               THREADS_PER_BLOCK, 0));
    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksDotProductPerSM, gpuDotProductsMerged, THREADS_PER_BLOCK, 0));
    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksCopyVectorPerSM, gpuCopyVector, THREADS_PER_BLOCK, 0));
    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksScaleVectorAndSaxpyPerSM, gpuScaleVectorAndSaxpy, THREADS_PER_BLOCK, 0));

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

        CUDA_RT_CALL(cudaStreamCreate(&streamsOtherOps[gpu_idx]));
        CUDA_RT_CALL(cudaStreamCreate(&streamsDot[gpu_idx]));
        CUDA_RT_CALL(cudaStreamCreate(&streamsSaxpy[gpu_idx]));
        CUDA_RT_CALL(cudaStreamCreate(&streamsSpMV[gpu_idx]));

        initVectors<<<initVectorsGridSize, THREADS_PER_BLOCK, 0, streamsOtherOps[gpu_idx]>>>(
            um_r, um_x, num_rows, gpu_idx, num_devices);

        gpuSpMV<<<spmvGridSize, THREADS_PER_BLOCK, 0, streamsOtherOps[gpu_idx]>>>(
            um_I, um_J, um_val, nnz, num_rows, float_positive_one, um_x, um_s, gpu_idx,
            num_devices);

        gpuSaxpy<<<saxpyGridSize, THREADS_PER_BLOCK, 0, streamsOtherOps[gpu_idx]>>>(
            um_s, um_r, um_float_negative_one, num_rows, gpu_idx, num_devices);

        // w0 = Ar0
        gpuSpMV<<<spmvGridSize, THREADS_PER_BLOCK, 0, streamsOtherOps[gpu_idx]>>>(
            um_I, um_J, um_val, nnz, num_rows, float_positive_one, um_r, um_w, gpu_idx,
            num_devices);

        gpuDotProductsMerged<<<dotProductGridSize, THREADS_PER_BLOCK, sMemSize,
                               streamsOtherOps[gpu_idx]>>>(um_r, um_r, um_r, um_w, num_rows,
                                                           gpu_idx, num_devices, sMemSize);

        addLocalDotContributions<<<1, 1, 0, streamsOtherOps[gpu_idx]>>>(um_dot_result_delta,
                                                                        um_dot_result_gamma);

        syncPeers<<<1, 1, 0, streamsOtherOps[gpu_idx]>>>(gpu_idx, num_devices,
                                                         hostMemoryArrivedList);

        um_tmp_dot_delta1[0] = um_dot_result_delta[0];

        int k = 1;

        while (k <= iter_max) {
            if (k > 1) {
                update_b_k<<<1, 1, 0, streamsOtherOps[gpu_idx]>>>(um_tmp_dot_delta1,
                                                                  um_tmp_dot_delta0, um_b);

                gpuScaleVectorAndSaxpy<<<scaleVectorAndSaxpyGridSize, THREADS_PER_BLOCK, 0,
                                         streamsOtherOps[gpu_idx]>>>(
                    um_r, um_p, float_positive_one, um_b, num_rows, gpu_idx, num_devices);

                gpuScaleVectorAndSaxpy<<<scaleVectorAndSaxpyGridSize, THREADS_PER_BLOCK, 0,
                                         streamsOtherOps[gpu_idx]>>>(
                    um_w, um_s, float_positive_one, um_b, num_rows, gpu_idx, num_devices);

                gpuScaleVectorAndSaxpy<<<scaleVectorAndSaxpyGridSize, THREADS_PER_BLOCK, 0,
                                         streamsOtherOps[gpu_idx]>>>(
                    um_t, um_u, float_positive_one, um_b, num_rows, gpu_idx, num_devices);
            } else {
                gpuCopyVector<<<copyVectorGridSize, THREADS_PER_BLOCK, 0,
                                streamsOtherOps[gpu_idx]>>>(um_r, um_p, num_rows, gpu_idx,
                                                            num_devices);

                // Need to figure out what to copy where
                // Other vectors also need to be initialized
                // Fine for now

                // gpuCopyVector<<<copyVectorGridSize, THREADS_PER_BLOCK, 0, nStreams[gpu_idx]>>>(
                //     um_s, um_w, num_rows, gpu_idx, num_devices);
            }

            syncPeers<<<1, 1, 0, 0>>>(gpu_idx, num_devices, hostMemoryArrivedList);

            // SpMV
            gpuSpMV<<<spmvGridSize, THREADS_PER_BLOCK, sMemSize, streamsSpMV[gpu_idx]>>>(
                um_I, um_J, um_val, nnz, num_rows, float_positive_one, um_p, um_s, gpu_idx,
                num_devices);

            // Two dot products => <r, r> and <r, w>
            resetLocalDotProducts<<<1, 1, 0, streamsDot[gpu_idx]>>>(um_dot_result_delta,
                                                                    um_dot_result_gamma);

            gpuDotProductsMerged<<<dotProductGridSize, THREADS_PER_BLOCK, sMemSize,
                                   streamsDot[gpu_idx]>>>(um_r, um_r, um_r, um_w, num_rows, gpu_idx,
                                                          num_devices, sMemSize);

            addLocalDotContributions<<<1, 1, 0, streamsDot[gpu_idx]>>>(um_dot_result_delta,
                                                                       um_dot_result_gamma);

            syncPeers<<<1, 1, 0, 0>>>(gpu_idx, num_devices, hostMemoryArrivedList);

            update_a_k<<<1, 1, 0, streamsSaxpy[gpu_idx]>>>(
                (float *)um_dot_result_delta, (float *)um_dot_result_gamma, um_b, um_a);

            gpuSaxpy<<<saxpyGridSize, THREADS_PER_BLOCK, 0, streamsSaxpy[gpu_idx]>>>(
                um_p, um_x, um_a, num_rows, gpu_idx, num_devices);

            a_minus<<<1, 1, 0, streamsSaxpy[gpu_idx]>>>(um_a, um_na);

            gpuSaxpy<<<saxpyGridSize, THREADS_PER_BLOCK, 0, streamsSaxpy[gpu_idx]>>>(
                um_s, um_r, um_na, num_rows, gpu_idx, num_devices);

            gpuSaxpy<<<saxpyGridSize, THREADS_PER_BLOCK, 0, streamsSaxpy[gpu_idx]>>>(
                um_u, um_w, um_na, num_rows, gpu_idx, num_devices);

            um_tmp_dot_delta0[0] = um_tmp_dot_delta1[0];
            um_tmp_dot_gamma0[0] = um_tmp_dot_gamma1[0];

            um_tmp_dot_delta1[0] = um_dot_result_delta[0];
            um_tmp_dot_gamma1[0] = um_dot_result_gamma[0];

            CUDA_RT_CALL(cudaStreamSynchronize(0));

#pragma omp barrier

            k++;
        }
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
                           stop, compare_to_single_gpu, tol);
        }
    }

    printf("Execution time: %8.4f s\n", (stop - start));

#if ENABLE_CPU_DEBUG_CODE
    cpuConjugateGrad(I, J, val, x_cpu, s_cpu, p_cpu, r_cpu, nz, N, tol);
#endif

    float rsum, diff, err = 0.0;

    for (int i = 0; i < num_rows; i++) {
        rsum = 0.0;

        for (int j = um_I[i]; j < um_J[i + 1]; j++) {
            rsum += host_val[j] * um_x[um_J[j]];
        }

        diff = fabs(rsum - rhs);

        if (diff > err) {
            err = diff;
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

#if ENABLE_CPU_DEBUG_CODE
    free(s_cpu);
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
