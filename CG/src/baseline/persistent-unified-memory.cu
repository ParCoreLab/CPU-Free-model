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

#include "../../include/baseline/persistent-unified-memory.cuh"
#include "../../include/common.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace BaselinePersistentUnifiedMemory {
const char *sSDKname = "conjugateGradientMultiDeviceCG";

#define ENABLE_CPU_DEBUG_CODE 0
#define THREADS_PER_BLOCK 512

__device__ double grid_dot_result = 0.0;

// Data filled on CPU needed for MultiGPU operations.
struct MultiDeviceData {
    unsigned char *hostMemoryArrivedList;
    unsigned int numDevices;
    unsigned int deviceRank;
};

// Class used for coordination of multiple devices.
class PeerGroup {
    const MultiDeviceData &data;
    const cg::grid_group &grid;

    __device__ unsigned char load_arrived(unsigned char *arrived) const {
#if __CUDA_ARCH__ < 700
        return *(volatile unsigned char *)arrived;
#else
        unsigned int result;
        asm volatile("ld.acquire.sys.global.u8 %0, [%1];" : "=r"(result) : "l"(arrived) : "memory");
        return result;
#endif
    }

    __device__ void store_arrived(unsigned char *arrived, unsigned char val) const {
#if __CUDA_ARCH__ < 700
        *(volatile unsigned char *)arrived = val;
#else
        unsigned int reg_val = val;
        asm volatile("st.release.sys.global.u8 [%1], %0;" ::"r"(reg_val) "l"(arrived) : "memory");

        // Avoids compiler warnings from unused variable val.
        (void)(reg_val = reg_val);
#endif
    }

   public:
    __device__ PeerGroup(const MultiDeviceData &data, const cg::grid_group &grid)
        : data(data), grid(grid){};

    __device__ unsigned int size() const { return data.numDevices * grid.size(); }

    __device__ unsigned int thread_rank() const {
        return data.deviceRank * grid.size() + grid.thread_rank();
    }

    __device__ void sync() const {
        grid.sync();

        // One thread from each grid participates in the sync.
        if (grid.thread_rank() == 0) {
            if (data.deviceRank == 0) {
                // Leader grid waits for others to join and then releases them.
                // Other GPUs can arrive in any order, so the leader have to wait for
                // all others.
                for (int i = 0; i < data.numDevices - 1; i++) {
                    while (load_arrived(&data.hostMemoryArrivedList[i]) == 0)
                        ;
                }
                for (int i = 0; i < data.numDevices - 1; i++) {
                    store_arrived(&data.hostMemoryArrivedList[i], 0);
                }
                __threadfence_system();
            } else {
                // Other grids note their arrival and wait to be released.
                store_arrived(&data.hostMemoryArrivedList[data.deviceRank - 1], 1);
                while (load_arrived(&data.hostMemoryArrivedList[data.deviceRank - 1]) == 1)
                    ;
            }
        }

        grid.sync();
    }
};

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

extern "C" __global__ void multiGpuConjugateGradient(int *I, int *J, float *val, float *x,
                                                     float *Ax, float *p, float *r,
                                                     double *dot_result, int nnz, int N, float tol,
                                                     MultiDeviceData multi_device_data,
                                                     const int iter_max) {
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

    gpuDotProduct(r, r, N, cta, peer_group);

    cg::sync(grid);

    if (grid.thread_rank() == 0) {
        atomicAdd_system(dot_result, grid_dot_result);
        grid_dot_result = 0.0;
    }
    peer_group.sync();

    r1 = *dot_result;

    int k = 1;

    // while (r1 > tol * tol && k <= iter_max)

    while (k <= iter_max) {
        // Saxpy 1 Start

        if (k > 1) {
            b = r1 / r0;
            gpuScaleVectorAndSaxpy(r, p, alpha, b, N, peer_group);
        } else {
            gpuCopyVector(r, p, N, peer_group);
        }

        peer_group.sync();

        // Saxpy 1 End

        // SpMV Start

        gpuSpMV(I, J, val, nnz, N, alpha, p, Ax, peer_group);

        // SpMV End

        // Dot Product 1 Start

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

        // Dot Product 1 End

        // Saxpy 2 Start

        a = r1 / *dot_result;

        gpuSaxpy(p, x, a, N, peer_group);

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

        gpuDotProduct(r, r, N, cta, peer_group);

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
}

// Map of device version to device number
std::multimap<std::pair<int, int>, int> getIdenticalGPUs() {
    int numGpus = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&numGpus));

    std::multimap<std::pair<int, int>, int> identicalGpus;

    for (int i = 0; i < numGpus; i++) {
        cudaDeviceProp deviceProp;
        CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, i));

        // Filter unsupported devices
        if (deviceProp.cooperativeLaunch && deviceProp.concurrentManagedAccess) {
            identicalGpus.emplace(std::make_pair(deviceProp.major, deviceProp.minor), i);
        }
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", i, deviceProp.name,
               deviceProp.major, deviceProp.minor);
    }

    return identicalGpus;
}
}  // namespace BaselinePersistentUnifiedMemory

int BaselinePersistentUnifiedMemory::init(int argc, char *argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 10000);
    std::string matrix_path = get_argval<std::string>(argv, argv + argc, "-matrix_path", "");

    int num_devices = 0;

    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

    int N = 0, nz = 0, *I = NULL, *J = NULL;
    float *val = NULL;
    const float tol = 1e-5f;
    float *x;
    float rhs = 1.0;
    float r1;
    float *r, *p, *Ax;

    printf("Starting [%s]...\n", BaselinePersistentUnifiedMemory::sSDKname);
    auto gpusByArch = getIdenticalGPUs();

    auto it = gpusByArch.begin();
    auto end = gpusByArch.end();

    auto bestFit = std::make_pair(it, it);
    // use std::distance to find the largest number of GPUs amongst architectures
    auto distance = [](decltype(bestFit) p) { return std::distance(p.first, p.second); };

    // Read each unique key/pair element in order
    for (; it != end; it = gpusByArch.upper_bound(it->first)) {
        // first and second are iterators bounded within the architecture group
        auto testFit = gpusByArch.equal_range(it->first);
        // Always use devices with highest architecture version or whichever has the
        // most devices available
        if (distance(bestFit) <= distance(testFit)) bestFit = testFit;
    }

    if (distance(bestFit) < num_devices) {
        printf(
            "No two or more GPUs with same architecture capable of "
            "concurrentManagedAccess found. "
            "\nWaiving the sample\n");
    }

    std::set<int> bestFitDeviceIds;

    // Check & select peer-to-peer access capable GPU devices as enabling p2p
    // access between participating GPUs gives better performance.
    for (auto itr = bestFit.first; itr != bestFit.second; itr++) {
        int deviceId = itr->second;
        CUDA_RT_CALL(cudaSetDevice(deviceId));

        std::for_each(
            itr, bestFit.second,
            [&deviceId, &bestFitDeviceIds, &num_devices](decltype(*itr) mapPair) {
                if (deviceId != mapPair.second) {
                    int access = 0;
                    CUDA_RT_CALL(cudaDeviceCanAccessPeer(&access, deviceId, mapPair.second));
                    printf("Device=%d %s Access Peer Device=%d\n", deviceId,
                           access ? "CAN" : "CANNOT", mapPair.second);
                    if (access && bestFitDeviceIds.size() < num_devices) {
                        bestFitDeviceIds.emplace(deviceId);
                        bestFitDeviceIds.emplace(mapPair.second);
                    } else {
                        printf("Ignoring device %i (max devices exceeded)\n", mapPair.second);
                    }
                }
            });

        if (bestFitDeviceIds.size() >= num_devices) {
            printf("Selected p2p capable devices - ");
            for (auto devicesItr = bestFitDeviceIds.begin(); devicesItr != bestFitDeviceIds.end();
                 devicesItr++) {
                printf("deviceId = %d  ", *devicesItr);
            }
            printf("\n");
            break;
        }
    }

    // if bestFitDeviceIds.size() == 0 it means the GPUs in system are not p2p
    // capable, hence we add it without p2p capability check.
    if (!bestFitDeviceIds.size()) {
        printf("Devices involved are not p2p capable.. selecting %zu of them\n", num_devices);
        std::for_each(bestFit.first, bestFit.second,
                      [&bestFitDeviceIds, &num_devices](decltype(*bestFit.first) mapPair) {
                          if (bestFitDeviceIds.size() < num_devices) {
                              bestFitDeviceIds.emplace(mapPair.second);
                          } else {
                              printf("Ignoring device %i (max devices exceeded)\n", mapPair.second);
                          }
                          // Insert the sequence into the deviceIds set
                      });
    } else {
        // perform cudaDeviceEnablePeerAccess in both directions for all
        // participating devices.
        for (auto p1_itr = bestFitDeviceIds.begin(); p1_itr != bestFitDeviceIds.end(); p1_itr++) {
            CUDA_RT_CALL(cudaSetDevice(*p1_itr));
            for (auto p2_itr = bestFitDeviceIds.begin(); p2_itr != bestFitDeviceIds.end();
                 p2_itr++) {
                if (*p1_itr != *p2_itr) {
                    CUDA_RT_CALL(cudaDeviceEnablePeerAccess(*p2_itr, 0));
                    CUDA_RT_CALL(cudaSetDevice(*p1_itr));
                }
            }
        }
    }

    /* Generate a random tridiagonal symmetric matrix in CSR format */
    N = 10485760 * 2;
    nz = (N - 2) * 3 + 4;

    CUDA_RT_CALL(cudaMallocManaged((void **)&I, sizeof(int) * (N + 1)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&J, sizeof(int) * nz));
    CUDA_RT_CALL(cudaMallocManaged((void **)&val, sizeof(float) * nz));

    float *val_cpu = (float *)malloc(sizeof(float) * nz);

    genTridiag(I, J, val_cpu, N, nz);

    memcpy(val, val_cpu, sizeof(float) * nz);
    CUDA_RT_CALL(cudaMemAdvise(I, sizeof(int) * (N + 1), cudaMemAdviseSetReadMostly, 0));
    CUDA_RT_CALL(cudaMemAdvise(J, sizeof(int) * nz, cudaMemAdviseSetReadMostly, 0));
    CUDA_RT_CALL(cudaMemAdvise(val, sizeof(float) * nz, cudaMemAdviseSetReadMostly, 0));

    CUDA_RT_CALL(cudaMallocManaged((void **)&x, sizeof(float) * N));

    double *dot_result;
    CUDA_RT_CALL(cudaMallocManaged((void **)&dot_result, sizeof(double)));

    CUDA_RT_CALL(cudaMemset(dot_result, 0, sizeof(double)));

    // temp memory for ConjugateGradient
    CUDA_RT_CALL(cudaMallocManaged((void **)&r, N * sizeof(float)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&p, N * sizeof(float)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&Ax, N * sizeof(float)));

    std::cout << "\nRunning on GPUs = " << num_devices << std::endl;
    cudaStream_t nStreams[num_devices];

    int sMemSize = sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1);
    int numBlocksPerSm = INT_MAX;
    int numThreads = THREADS_PER_BLOCK;
    int numSms = INT_MAX;
    auto deviceId = bestFitDeviceIds.begin();

    // set numSms & numBlocksPerSm to be lowest of 2 devices
    while (deviceId != bestFitDeviceIds.end()) {
        cudaDeviceProp deviceProp;
        CUDA_RT_CALL(cudaSetDevice(*deviceId));
        CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, *deviceId));

        int numBlocksPerSm_current = 0;
        CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocksPerSm_current, multiGpuConjugateGradient, numThreads, sMemSize));

        if (numBlocksPerSm > numBlocksPerSm_current) {
            numBlocksPerSm = numBlocksPerSm_current;
        }
        if (numSms > deviceProp.multiProcessorCount) {
            numSms = deviceProp.multiProcessorCount;
        }
        deviceId++;
    }

    // Added this line to time the different kernel operations
    numBlocksPerSm = 2;

    if (!numBlocksPerSm) {
        printf(
            "Max active blocks per SM is returned as 0.\n Hence, Waiving the "
            "sample\n");
    }

    int device_count = 0;
    int totalThreadsPerGPU = numSms * numBlocksPerSm * THREADS_PER_BLOCK;
    deviceId = bestFitDeviceIds.begin();
    while (deviceId != bestFitDeviceIds.end()) {
        CUDA_RT_CALL(cudaSetDevice(*deviceId));
        CUDA_RT_CALL(cudaStreamCreate(&nStreams[device_count]));

        int perGPUIter = N / (totalThreadsPerGPU * num_devices);
        int offset_Ax = device_count * totalThreadsPerGPU;
        int offset_r = device_count * totalThreadsPerGPU;
        int offset_p = device_count * totalThreadsPerGPU;
        int offset_x = device_count * totalThreadsPerGPU;

        CUDA_RT_CALL(cudaMemPrefetchAsync(I, sizeof(int) * N, *deviceId, nStreams[device_count]));
        CUDA_RT_CALL(
            cudaMemPrefetchAsync(val, sizeof(float) * nz, *deviceId, nStreams[device_count]));
        CUDA_RT_CALL(
            cudaMemPrefetchAsync(J, sizeof(float) * nz, *deviceId, nStreams[device_count]));

        if (offset_Ax <= N) {
            for (int i = 0; i < perGPUIter; i++) {
                cudaMemAdvise(Ax + offset_Ax, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetPreferredLocation, *deviceId);
                cudaMemAdvise(r + offset_r, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetPreferredLocation, *deviceId);
                cudaMemAdvise(x + offset_x, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetPreferredLocation, *deviceId);
                cudaMemAdvise(p + offset_p, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetPreferredLocation, *deviceId);

                cudaMemAdvise(Ax + offset_Ax, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetAccessedBy, *deviceId);
                cudaMemAdvise(r + offset_r, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetAccessedBy, *deviceId);
                cudaMemAdvise(p + offset_p, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetAccessedBy, *deviceId);
                cudaMemAdvise(x + offset_x, sizeof(float) * totalThreadsPerGPU,
                              cudaMemAdviseSetAccessedBy, *deviceId);

                offset_Ax += totalThreadsPerGPU * num_devices;
                offset_r += totalThreadsPerGPU * num_devices;
                offset_p += totalThreadsPerGPU * num_devices;
                offset_x += totalThreadsPerGPU * num_devices;

                if (offset_Ax >= N) {
                    break;
                }
            }
        }

        device_count++;
        deviceId++;
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

    printf("Total threads per GPU = %d numBlocksPerSm  = %d\n",
           numSms * numBlocksPerSm * THREADS_PER_BLOCK, numBlocksPerSm);
    dim3 dimGrid(numSms * numBlocksPerSm, 1, 1), dimBlock(THREADS_PER_BLOCK, 1, 1);

    // Structure used for cross-grid synchronization.
    BaselinePersistentUnifiedMemory::MultiDeviceData multi_device_data;
    CUDA_RT_CALL(cudaHostAlloc(&multi_device_data.hostMemoryArrivedList,
                               (num_devices - 1) * sizeof(*multi_device_data.hostMemoryArrivedList),
                               cudaHostAllocPortable));
    memset(multi_device_data.hostMemoryArrivedList, 0,
           (num_devices - 1) * sizeof(*multi_device_data.hostMemoryArrivedList));
    multi_device_data.numDevices = num_devices;
    multi_device_data.deviceRank = 0;

    void *kernelArgs[] = {(void *)&I,       (void *)&J, (void *)&val, (void *)&x,
                          (void *)&Ax,      (void *)&p, (void *)&r,   (void *)&dot_result,
                          (void *)&nz,      (void *)&N, (void *)&tol, (void *)&multi_device_data,
                          (void *)&iter_max};

    printf("Launching kernel\n");

    deviceId = bestFitDeviceIds.begin();
    device_count = 0;

    double start = omp_get_wtime();

    while (deviceId != bestFitDeviceIds.end()) {
        CUDA_RT_CALL(cudaSetDevice(*deviceId));
        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *)multiGpuConjugateGradient, dimGrid,
                                                 dimBlock, kernelArgs, sMemSize,
                                                 nStreams[device_count++]));
        multi_device_data.deviceRank++;
        deviceId++;
    }

    CUDA_RT_CALL(cudaMemPrefetchAsync(x, sizeof(float) * N, cudaCpuDeviceId));
    CUDA_RT_CALL(cudaMemPrefetchAsync(dot_result, sizeof(double), cudaCpuDeviceId));

    deviceId = bestFitDeviceIds.begin();
    device_count = 0;
    while (deviceId != bestFitDeviceIds.end()) {
        CUDA_RT_CALL(cudaSetDevice(*deviceId));
        CUDA_RT_CALL(cudaStreamSynchronize(nStreams[device_count++]));
        deviceId++;
    }

    r1 = (float)*dot_result;

    double stop = omp_get_wtime();

    printf("Execution time: %8.4f s\n", (stop - start));

    printf("GPU Final, residual = %e \n  ", sqrt(r1));

#if ENABLE_CPU_DEBUG_CODE
    cpuConjugateGrad(I, J, val, x_cpu, Ax_cpu, p_cpu, r_cpu, nz, N, tol);
#endif

    float rsum, diff, err = 0.0;

    for (int i = 0; i < N; i++) {
        rsum = 0.0;

        for (int j = I[i]; j < I[i + 1]; j++) {
            rsum += val_cpu[j] * x[J[j]];
        }

        diff = fabs(rsum - rhs);

        if (diff > err) {
            err = diff;
        }
    }

    CUDA_RT_CALL(cudaFreeHost(multi_device_data.hostMemoryArrivedList));
    CUDA_RT_CALL(cudaFree(I));
    CUDA_RT_CALL(cudaFree(J));
    CUDA_RT_CALL(cudaFree(val));
    CUDA_RT_CALL(cudaFree(x));
    CUDA_RT_CALL(cudaFree(r));
    CUDA_RT_CALL(cudaFree(p));
    CUDA_RT_CALL(cudaFree(Ax));
    CUDA_RT_CALL(cudaFree(dot_result));
    free(val_cpu);

#if ENABLE_CPU_DEBUG_CODE
    free(Ax_cpu);
    free(r_cpu);
    free(p_cpu);
    free(x_cpu);
#endif

    printf("Test Summary:  Error amount = %f \n", err);
    fprintf(stdout, "&&&& conjugateGradientMultiDeviceCG %s\n",
            (sqrt(r1) < tol) ? "PASSED" : "FAILED");
    exit((sqrt(r1) < tol) ? EXIT_SUCCESS : EXIT_FAILURE);
}