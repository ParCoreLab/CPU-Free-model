/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <sstream>

#include "../include/common.h"
#include "../include/single-gpu-naive.cuh"

#ifdef USE_NVTX
#include <nvToolsExt.h>

const uint32_t colors[] = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff,
                           0x0000ffff, 0x00ff0000, 0x00ffffff};
const int num_colors = sizeof(colors) / sizeof(uint32_t);

#define PUSH_RANGE(name, cid)                              \
    {                                                      \
        int color_id = cid;                                \
        color_id = color_id % num_colors;                  \
        nvtxEventAttributes_t eventAttrib = {0};           \
        eventAttrib.version = NVTX_VERSION;                \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;  \
        eventAttrib.colorType = NVTX_COLOR_ARGB;           \
        eventAttrib.color = colors[color_id];              \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name;                  \
        nvtxRangePushEx(&eventAttrib);                     \
    }
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name, cid)
#define POP_RANGE
#endif

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace SingleGPUNaive {
    __global__ void initialize_boundaries(real *__restrict__ const a_new, real *__restrict__ const a,
                                          const real pi, const int nx, const int ny) {
        for (int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < ny; iy += blockDim.x * gridDim.x) {
            const real y0 = sin(2.0 * pi * iy / (ny - 1));
            a[iy * nx + 0] = y0;
            a[iy * nx + (nx - 1)] = y0;
            a_new[iy * nx + 0] = y0;
            a_new[iy * nx + (nx - 1)] = y0;
        }
    }

    __global__ void jacobi_kernel(real *__restrict__ a_new, const real *__restrict__ a,
                                  const int iy_start, const int iy_end, const int nx, const int niter) {
        cg::thread_block cta = cg::this_thread_block();
        cg::grid_group grid = cg::this_grid();

        const int iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
        const int ix = blockIdx.x * blockDim.x + threadIdx.x;

        real local_l2_norm = 0.0;

        int i = 0;

        while (i < niter) {
            if (iy < iy_end) {
                if (ix >= 1 && ix < (nx - 1)) {
                    const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                                 a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
                    a_new[iy * nx + ix] = new_val;

                    // apply boundary conditions
                    if (iy_start == iy) {
                        a_new[iy_end * nx + ix] = new_val;
                    }

                    if ((iy_end - 1) == iy) {
                        a_new[(iy_start - 1) * nx + ix] = new_val;
                    }

                    real residue = new_val - a[iy * nx + ix];
                    local_l2_norm = residue * residue;
                }
            }

            real *temp_pointer = a_new;
            a = a_new;
            a_new = temp_pointer;

            i++;
            grid.sync();
        }
    }
}  // namespace SingleGPUNaive

struct l2_norm_buf {
    cudaEvent_t copy_done;
    real *d;
    real *h;
};

int SingleGPUNaive::init(int argc, char *argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nccheck = get_argval<int>(argv, argv + argc, "-nccheck", 1);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);
    const bool csv = get_arg(argv, argv + argc, "-csv");

    if (nccheck != 1) {
        fprintf(stderr, "Only nccheck = 1 is supported\n");
        return -1;
    }

    real *a;
    real *a_new;

    cudaStream_t compute_stream;
    cudaStream_t copy_l2_norm_stream;
    cudaStream_t reset_l2_norm_stream;

    cudaEvent_t compute_done;
    cudaEvent_t reset_l2_norm_done[2];

    real l2_norms[2];
    l2_norm_buf l2_norm_bufs[2];

    int iy_start = 1;
    int iy_end = (ny - 1);

    CUDA_RT_CALL(cudaSetDevice(0));
    CUDA_RT_CALL(cudaFree(0));

    CUDA_RT_CALL(cudaMalloc(&a, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc(&a_new, nx * ny * sizeof(real)));

    CUDA_RT_CALL(cudaMemset(a, 0, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * ny * sizeof(real)));

    // Set diriclet boundary conditions on left and right boarder
    SingleGPUNaive::initialize_boundaries<<<ny / 128 + 1, 128>>>(a, a_new, PI, nx, ny);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());

    //    CUDA_RT_CALL(cudaStreamCreate(&compute_stream));
    //    CUDA_RT_CALL(cudaStreamCreate(&copy_l2_norm_stream));
    //    CUDA_RT_CALL(cudaStreamCreate(&reset_l2_norm_stream));
    //    CUDA_RT_CALL(cudaEventCreateWithFlags(&compute_done, cudaEventDisableTiming));
    //    CUDA_RT_CALL(cudaEventCreateWithFlags(&reset_l2_norm_done[0], cudaEventDisableTiming));
    //    CUDA_RT_CALL(cudaEventCreateWithFlags(&reset_l2_norm_done[1], cudaEventDisableTiming));

    //    for (int i = 0; i < 2; ++i) {
    //        CUDA_RT_CALL(cudaEventCreateWithFlags(&l2_norm_bufs[i].copy_done,
    //        cudaEventDisableTiming)); CUDA_RT_CALL(cudaMalloc(&l2_norm_bufs[i].d, sizeof(real)));
    //        CUDA_RT_CALL(cudaMemset(l2_norm_bufs[i].d, 0, sizeof(real)));
    //        CUDA_RT_CALL(cudaMallocHost(&l2_norm_bufs[i].h, sizeof(real)));
    //        (*l2_norm_bufs[i].h) = 1.0;
    //    }

    CUDA_RT_CALL(cudaDeviceSynchronize());

    if (!csv)
        printf(
                "Jacobi relaxation: %d iterations on %d x %d mesh with norm check "
                "every %d iterations\n",
                iter_max, ny, nx, nccheck);

    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 32;
    dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x, (ny + dim_block_y - 1) / dim_block_y, 1);

    int iter = 0;
    for (int i = 0; i < 2; ++i) {
        l2_norms[i] = 0.0;
    }

    double start = omp_get_wtime();

    PUSH_RANGE("Jacobi solve", 0)

    bool l2_norm_greater_than_tol = true;
    void *kernelArgs[] = {
            (void *) &a_new,
            (void *) &a,
            //        (void *)&l2_norm_bufs[curr].d,
            (void *) &iy_start,
            (void *) &iy_end,
            (void *) &nx,
            (void *) &iter_max,
    };

    // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp{};
    int devID = 0;  // findCudaDevice(argc, (const char **)argv);
    CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, devID));
    int numSms = deviceProp.multiProcessorCount;

    constexpr int THREADS_PER_BLOCK = 512;

    int sMemSize = sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1);
    int numBlocksPerSm = 0;
    int numThreads = THREADS_PER_BLOCK;

    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocksPerSm, SingleGPUNaive::jacobi_kernel, numThreads, 0));

    // This is stupid
    int blocks_each = (int) sqrt(numSms * numBlocksPerSm);
    int threads_each = (int) sqrt(THREADS_PER_BLOCK);
    dim3 dimGrid(blocks_each, blocks_each), dimBlock(threads_each, threads_each);

    //   dim3 threads(2, 2);
    //   dim3 blocks(5, 5);

    CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *) SingleGPUNaive::jacobi_kernel, dimGrid,
                                             dimBlock, kernelArgs, 0, nullptr));

    CUDA_RT_CALL(cudaDeviceSynchronize());

    //    cudaMemcpy()

    //    return;
    //
    //    while (l2_norm_greater_than_tol && iter < iter_max) {
    //        // on new iteration: old current vars are now previous vars, old
    //        // previous vars are no longer needed
    //        int prev = iter % 2;
    //        int curr = (iter + 1) % 2;
    //
    //        // wait for memset from old previous iteration to complete
    //        CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, reset_l2_norm_done[curr], 0));
    //
    //        jacobi_kernel<dim_block_x, dim_block_y>
    //            <<<dim_grid, {dim_block_x, dim_block_y, 1}, 0, compute_stream>>>(
    //                a_new, a, l2_norm_bufs[curr].d, iy_start, iy_end, nx);
    //        CUDA_RT_CALL(cudaGetLastError());
    //        CUDA_RT_CALL(cudaEventRecord(compute_done, compute_stream));
    //
    //        // perform L2 norm calculation
    //        if ((iter % nccheck) == 0 || (!csv && (iter % 100) == 0)) {
    //            CUDA_RT_CALL(cudaStreamWaitEvent(copy_l2_norm_stream, compute_done, 0));
    //            CUDA_RT_CALL(cudaMemcpyAsync(l2_norm_bufs[curr].h, l2_norm_bufs[curr].d,
    //            sizeof(real),
    //                                         cudaMemcpyDeviceToHost, copy_l2_norm_stream));
    //            CUDA_RT_CALL(cudaEventRecord(l2_norm_bufs[curr].copy_done, copy_l2_norm_stream));
    //
    //            // make sure D2H copy is complete before using the data for
    //            // calculation
    //            CUDA_RT_CALL(cudaEventSynchronize(l2_norm_bufs[prev].copy_done));
    //
    //            l2_norms[prev] = *(l2_norm_bufs[prev].h);
    //            l2_norms[prev] = std::sqrt(l2_norms[prev]);
    //            l2_norm_greater_than_tol = (l2_norms[prev] > tol);
    //
    //            if (!csv && (iter % 100) == 0) {
    //                printf("%5d, %0.6f\n", iter, l2_norms[prev]);
    //            }
    //
    //            // reset everything for next iteration
    //            l2_norms[prev] = 0.0;
    //            *(l2_norm_bufs[prev].h) = 0.0;
    //            CUDA_RT_CALL(
    //                cudaMemsetAsync(l2_norm_bufs[prev].d, 0, sizeof(real), reset_l2_norm_stream));
    //            CUDA_RT_CALL(cudaEventRecord(reset_l2_norm_done[prev], reset_l2_norm_stream));
    //        }
    //
    //        std::swap(a_new, a);
    //        iter++;
    //    }
    //    CUDA_RT_CALL(cudaDeviceSynchronize());
    //    POP_RANGE
    //    double stop = omp_get_wtime();
    //
    //    if (csv) {
    //        printf("single_gpu, %d, %d, %d, %d, %f\n", nx, ny, iter_max, nccheck, (stop - start));
    //    } else {
    //        printf("%dx%d: 1 GPU: %8.4f s\n", ny, nx, (stop - start));
    //    }
    //
    //    for (int i = 0; i < 2; ++i) {
    //        CUDA_RT_CALL(cudaFreeHost(l2_norm_bufs[i].h));
    //        CUDA_RT_CALL(cudaFree(l2_norm_bufs[i].d));
    //        CUDA_RT_CALL(cudaEventDestroy(l2_norm_bufs[i].copy_done));
    //    }
    //
    //    CUDA_RT_CALL(cudaEventDestroy(reset_l2_norm_done[1]));
    //    CUDA_RT_CALL(cudaEventDestroy(reset_l2_norm_done[0]));
    //    CUDA_RT_CALL(cudaEventDestroy(compute_done));
    //
    //    CUDA_RT_CALL(cudaStreamDestroy(reset_l2_norm_stream));
    //    CUDA_RT_CALL(cudaStreamDestroy(copy_l2_norm_stream));
    //    CUDA_RT_CALL(cudaStreamDestroy(compute_stream));
    //
    //    CUDA_RT_CALL(cudaFree(a_new));
    //    CUDA_RT_CALL(cudaFree(a));
    //
    //    return 0;
}
