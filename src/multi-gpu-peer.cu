/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <sstream>

#include <omp.h>

#include "../include/single-gpu-naive.cuh"

//#ifdef HAVE_CUB
//#include <cub/block/block_reduce.cuh>
//#endif  // HAVE_CUB

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

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus)                                                      \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
    }

typedef float real;
constexpr real tol = 1.0e-8;

const real PI = 2.0 * std::asin(1.0);

__global__ void initialize_boundaries(real* __restrict__ const a_new, real* __restrict__ const a,
                                      const real pi, const int nx, const int ny) {
    for (int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < ny; iy += blockDim.x * gridDim.x) {
        const real y0 = sin(2.0 * pi * iy / (ny - 1));
        a[iy * nx + 0] = y0;
        a[iy * nx + (nx - 1)] = y0;
        a_new[iy * nx + 0] = y0;
        a_new[iy * nx + (nx - 1)] = y0;
    }
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void jacobi_kernel(real* __restrict__ a_new, const real* __restrict__ a,
                              const int iy_start, const int iy_end, const int nx, const int niter, int *flag) {
    //#ifdef HAVE_CUB
    //    typedef cub::BlockReduce<real, BLOCK_DIM_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS,
    //    BLOCK_DIM_Y>
    //        BlockReduce;
    //    __shared__ typename BlockReduce::TempStorage temp_storage;
    //#endif  // HAVE_CUB

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

        real* temp_pointer = a_new;
        a = a_new;
        a_new = temp_pointer;

        i++;
        grid.sync();
    }

    if (threadIdx.x == 0) {
        *flag = 1;
    }



    //#ifdef HAVE_CUB
    //    real block_l2_norm = BlockReduce(temp_storage).Sum(local_l2_norm);
    //    if (0 == threadIdx.y && 0 == threadIdx.x) atomicAdd(l2_norm, block_l2_norm);
    //#else
    //    atomicAdd(l2_norm, local_l2_norm);
    //#endif  // HAVE_CUB
}

__global__ void boundary_sync_kernel(real* __restrict__ a_new, int* flag) {
    while (!*flag) {

    }

//    printf("Sync\n");

//    *flag = 0;

//    if (threadIdx.x == 0) {
//        printf("Sync\n");
//    }
}

double noopt(const int nx, const int ny, const int iter_max, real* const a_ref_h, const int nccheck,
             const bool print);

template <typename T>
T get_argval(char** begin, char** end, const std::string& arg, const T default_val) {
    T argval = default_val;
    char** itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

bool get_arg(char** begin, char** end, const std::string& arg) {
    char** itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

struct l2_norm_buf {
    cudaEvent_t copy_done;
    real* d;
    real* h;
};

int init(int argc, char* argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nccheck = get_argval<int>(argv, argv + argc, "-nccheck", 1);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);
    const bool csv = get_arg(argv, argv + argc, "-csv");

    if (nccheck != 1) {
        fprintf(stderr, "Only nccheck = 1 is supported\n");
        return -1;
    }

    real* a;
    real* a_new;

//    cudaStream_t compute_stream;
//    cudaStream_t copy_l2_norm_stream;
//    cudaStream_t reset_l2_norm_stream;
//
//    cudaEvent_t compute_done;
//    cudaEvent_t reset_l2_norm_done[2];

    real l2_norms[2];
    l2_norm_buf l2_norm_bufs[2];

    int iy_start = 1;
    int iy_end = (ny - 1);

    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

#pragma omp parallel num_threads(num_devices)
    {
        int dev_id = omp_get_thread_num();

        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaFree(0));

        // For debugging locally
        if (num_devices > 1) {
            int canAccessPeer = 0;
            const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);

            CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, top));
            if (canAccessPeer) {
                CUDA_RT_CALL(cudaDeviceEnablePeerAccess(top, 0));
            } else {
                std::cerr << "P2P access required from " << dev_id << " to " << top << std::endl;
                std::exit(1);
            }
        }
    }

    CUDA_RT_CALL(cudaMalloc(&a, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc(&a_new, nx * ny * sizeof(real)));

    CUDA_RT_CALL(cudaMemset(a, 0, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * ny * sizeof(real)));

    // Set diriclet boundary conditions on left and right boarder
    initialize_boundaries<<<ny / 128 + 1, 128>>>(a, a_new, PI, nx, ny);
    CUDA_RT_CALL(cudaGetLastError());
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

    int *flag;
    CUDA_RT_CALL(cudaMalloc(&flag, 1 * sizeof(int)));
    CUDA_RT_CALL(cudaMemset(flag, 0, 1 * sizeof(int)));

    bool l2_norm_greater_than_tol = true;
    void* kernelArgs[] = {
        (void*)&a_new,
        (void*)&a,
        //        (void *)&l2_norm_bufs[curr].d,
        (void*)&iy_start,
        (void*)&iy_end,
        (void*)&nx,
        (void*)&iter_max,
        (void*)&flag
    };

    // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp{};
    int devID = 0;  // findCudaDevice(argc, (const char **)argv);
    CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, devID));
    int numSms = deviceProp.multiProcessorCount;

    constexpr int THREADS_PER_BLOCK = 1024;

    int sMemSize = sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1);
    int numBlocksPerSm = 0;
    int numThreads = THREADS_PER_BLOCK;

    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm, jacobi_kernel<dim_block_x, dim_block_y>, numThreads, 0));

//    numSms -= 1;

    // This is stupid
    int blocks_each = (int) sqrt(numSms * numBlocksPerSm);
    int threads_each = (int) sqrt(THREADS_PER_BLOCK);
    dim3 dimGrid(blocks_each, blocks_each), dimBlock(threads_each, threads_each);

    //   dim3 threads(2, 2);
    //   dim3 blocks(5, 5);

    int leastPriority = 0;
    int greatestPriority = leastPriority;
    CUDA_RT_CALL(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

#pragma omp parallel num_threads(num_devices)
    {
        // Add stream priority
        cudaStream_t inner_domain_stream;
        cudaStream_t boundary_sync_stream;

        CUDA_RT_CALL(cudaStreamCreateWithPriority(&inner_domain_stream, cudaStreamNonBlocking, greatestPriority));
        CUDA_RT_CALL(cudaStreamCreateWithPriority(&boundary_sync_stream, cudaStreamNonBlocking, leastPriority));

        int dev_id = omp_get_thread_num();
        CUDA_RT_CALL(cudaSetDevice(dev_id));

        // Inner domain
        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void*)jacobi_kernel<dim_block_x, dim_block_y>,
                                                 dimGrid, dimBlock, kernelArgs, 0, inner_domain_stream));

        // Boundary
        boundary_sync_kernel<<<1, 1, 0, boundary_sync_stream>>>(a, flag);

        CUDA_RT_CALL(cudaGetLastError());

        cudaDeviceSynchronize();
    }
}
