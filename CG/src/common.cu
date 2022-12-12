#include "../include/common.h"

#include <math.h>
#include <omp.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <iostream>

namespace cg = cooperative_groups;

bool get_arg(char **begin, char **end, const std::string &arg) {
    char **itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, real *val, int N, int nnz) {
    I[0] = 0, J[0] = 0, J[1] = 1;
    val[0] = (real)rand() / RAND_MAX + 10.0f;
    val[1] = (real)rand() / RAND_MAX;
    int start;

    for (int i = 1; i < N; i++) {
        if (i > 1) {
            I[i] = I[i - 1] + 3;
        } else {
            I[1] = 2;
        }

        start = (i - 1) * 3 + 2;
        J[start] = i - 1;
        J[start + 1] = i;

        if (i < N - 1) {
            J[start + 2] = i + 1;
        }

        val[start] = val[start - 1];
        val[start + 1] = (real)rand() / RAND_MAX + 10.0f;

        if (i < N - 1) {
            val[start + 2] = (real)rand() / RAND_MAX;
        }
    }

    I[N] = nnz;
}

void report_results(const int num_rows, real *x_ref_single_gpu, real *x_ref_cpu, real *x,
                    const int num_devices, const double single_gpu_runtime, const double start,
                    const double stop, const bool compare_to_single_gpu,
                    const bool compare_to_cpu) {
    bool result_correct_single_gpu = true;
    bool result_correct_cpu = true;

    int i = 0;

    if (compare_to_single_gpu) {
        std::cout << "Comparing correctness against Single GPU" << std::endl;

        for (i = 0; result_correct_single_gpu && (i < num_rows); i++) {
            if (std::fabs(x_ref_single_gpu[i] - x[i]) > tol || isnan(x[i]) ||
                isnan(x_ref_single_gpu[i])) {
                fprintf(stderr,
                        "ERROR: x[%d] = %.8f does not match %.8f "
                        "(reference)\n",
                        i, x[i], x_ref_single_gpu[i]);

                result_correct_single_gpu = false;
            }
        }
    }

    if (compare_to_cpu) {
        std::cout << "Comparing correctness against CPU" << std::endl;

        for (i = 0; result_correct_cpu && (i < num_rows); i++) {
            if (std::fabs(x_ref_cpu[i] - x[i]) > tol || isnan(x[i]) || isnan(x_ref_cpu[i])) {
                fprintf(stderr,
                        "ERROR: x[%d] = %.8f does not match %.8f "
                        "(reference)\n",
                        i, x[i], x_ref_cpu[i]);

                result_correct_cpu = false;
            }
        }
    }

    if (result_correct_single_gpu && result_correct_cpu) {
        printf("Execution time: %8.4f s\n", (stop - start));

        if (compare_to_single_gpu) {
            printf(
                "Non-persistent kernel - 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: %8.2f, "
                "efficiency: %8.2f \n",
                single_gpu_runtime, num_devices, (stop - start),
                single_gpu_runtime / (stop - start),
                single_gpu_runtime / (num_devices * (stop - start)) * 100);
        }
    }
}

// Common Single GPU kernels
namespace SingleGPU {
__global__ void initVectors(real *r, real *x, int num_rows) {
    size_t grid_size = gridDim.x * blockDim.x;
    size_t grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = grid_rank; i < num_rows; i += grid_size) {
        r[i] = 1.0;
        x[i] = 0.0;
    }
}

__global__ void gpuCopyVector(real *srcA, real *destB, int num_rows) {
    size_t grid_size = gridDim.x * blockDim.x;
    size_t grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = grid_rank; i < num_rows; i += grid_size) {
        destB[i] = srcA[i];
    }
}

__global__ void gpuSpMV(int *I, int *J, real *val, int nnz, int num_rows, real alpha,
                        real *inputVecX, real *outputVecY) {
    size_t grid_size = gridDim.x * blockDim.x;
    size_t grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = grid_rank; i < num_rows; i += grid_size) {
        int row_elem = I[i];
        int next_row_elem = I[i + 1];
        int num_elems_this_row = next_row_elem - row_elem;

        real output = 0.0;
        for (int j = 0; j < num_elems_this_row; j++) {
            output += alpha * val[row_elem + j] * inputVecX[J[row_elem + j]];
        }

        outputVecY[i] = output;
    }
}

__global__ void gpuSaxpy(real *x, real *y, real a, int num_rows) {
    size_t grid_size = gridDim.x * blockDim.x;
    size_t grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = grid_rank; i < num_rows; i += grid_size) {
        y[i] = a * x[i] + y[i];
    }
}

__global__ void gpuScaleVectorAndSaxpy(real *x, real *y, real a, real scale, int num_rows) {
    size_t grid_size = gridDim.x * blockDim.x;
    size_t grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = grid_rank; i < num_rows; i += grid_size) {
        y[i] = a * x[i] + scale * y[i];
    }
}

__global__ void r1_div_x(real r1, real r0, real *b) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        *b = r1 / r0;
    }
}

__global__ void a_minus(real a, real *na) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        *na = -a;
    }
}

__global__ void update_a_k(real dot_delta_1, real dot_gamma_1, real b, real *a) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        *a = dot_delta_1 / (dot_gamma_1 - (b / *a) * dot_delta_1);
    }
}

__global__ void update_b_k(real dot_delta_1, real dot_delta_0, real *b) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        *b = dot_delta_1 / dot_delta_0;
    }
}

__global__ void init_a_k(real dot_delta_1, real dot_gamma_1, real *a) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        *a = dot_delta_1 / dot_gamma_1;
    }
}

__global__ void init_b_k(real *b) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        *b = 0.0;
    }
}
}  // namespace SingleGPU

// Common Multi GPU kernels
namespace MultiGPU {
__global__ void initVectors(real *r, real *x, int num_rows, const int device_rank,
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

__global__ void gpuSpMV(int *I, int *J, real *val, int nnz, int num_rows, real alpha,
                        real *inputVecX, real *outputVecY, const int device_rank,
                        const int num_devices) {
    size_t local_grid_size = gridDim.x * blockDim.x;
    size_t local_grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    size_t global_grid_size = local_grid_size * num_devices;
    size_t global_grid_rank = device_rank * local_grid_size + local_grid_rank;

    for (size_t i = global_grid_rank; i < num_rows; i += global_grid_size) {
        int row_elem = I[i];
        int next_row_elem = I[i + 1];
        int num_elems_this_row = next_row_elem - row_elem;

        real output = 0.0;
        for (int j = 0; j < num_elems_this_row; j++) {
            output += alpha * val[row_elem + j] * inputVecX[J[row_elem + j]];
        }

        outputVecY[i] = output;
    }
}

__global__ void gpuSaxpy(real *x, real *y, real a, int num_rows, const int device_rank,
                         const int num_devices) {
    size_t local_grid_size = gridDim.x * blockDim.x;
    size_t local_grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    size_t global_grid_size = local_grid_size * num_devices;
    size_t global_grid_rank = device_rank * local_grid_size + local_grid_rank;

    for (size_t i = global_grid_rank; i < num_rows; i += global_grid_size) {
        y[i] = a * x[i] + y[i];
    }
}

__global__ void gpuCopyVector(real *srcA, real *destB, int num_rows, const int device_rank,
                              const int num_devices) {
    size_t local_grid_size = gridDim.x * blockDim.x;
    size_t local_grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    size_t global_grid_size = local_grid_size * num_devices;
    size_t global_grid_rank = device_rank * local_grid_size + local_grid_rank;

    for (size_t i = global_grid_rank; i < num_rows; i += global_grid_size) {
        destB[i] = srcA[i];
    }
}

__global__ void gpuScaleVectorAndSaxpy(real *x, real *y, real a, real scale, int num_rows,
                                       const int device_rank, const int num_devices) {
    size_t local_grid_size = gridDim.x * blockDim.x;
    size_t local_grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    size_t global_grid_size = local_grid_size * num_devices;
    size_t global_grid_rank = device_rank * local_grid_size + local_grid_rank;

    for (size_t i = global_grid_rank; i < num_rows; i += global_grid_size) {
        y[i] = a * x[i] + scale * y[i];
    }
}

// Multi-GPU Sync Kernel

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

__global__ void r1_div_x(real r1, real r0, real *b, const int gpu_idx) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gpu_idx == 0 && gid == 0) {
        *b = r1 / r0;
    }
}

__global__ void a_minus(real a, real *na, const int gpu_idx) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gpu_idx == 0 && gid == 0) {
        *na = -a;
    }
}

__global__ void update_a_k(real dot_delta_1, real dot_gamma_1, real b, real *a, const int gpu_idx) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gpu_idx == 0 && gid == 0) {
        *a = dot_delta_1 / (dot_gamma_1 - (b / *a) * dot_delta_1);
    }
}

__global__ void update_b_k(real dot_delta_1, real dot_delta_0, real *b, const int gpu_idx) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gpu_idx == 0 && gid == 0) {
        *b = dot_delta_1 / dot_delta_0;
    }
}

__global__ void init_a_k(real dot_delta_1, real dot_gamma_1, real *a, const int gpu_idx) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gpu_idx == 0 && gid == 0) {
        *a = dot_delta_1 / dot_gamma_1;
    }
}

__global__ void init_b_k(real *b, const int gpu_idx) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gpu_idx == 0 && gid == 0) {
        *b = 0.0;
    }
}
}  // namespace MultiGPU

// Single GPU Pipelined Implementation

namespace SingleGPUPipelinedDiscrete {
__device__ double grid_dot_result_delta = 0.0;
__device__ double grid_dot_result_gamma = 0.0;

// Performs two dot products at the same time
// Used to perform <r, r> and <r, w> at the same time
// Can we combined the two atomicAdds somehow?

__global__ void gpuDotProductsMerged(real *vecA_delta, real *vecB_delta, real *vecA_gamma,
                                     real *vecB_gamma, int num_rows, const int sMemSize) {
    cg::thread_block cta = cg::this_thread_block();

    size_t grid_size = gridDim.x * blockDim.x;
    size_t grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double tmp[];

    double *tmp_delta = (double *)tmp;
    double *tmp_gamma = (double *)&tmp_delta[sMemSize / (2 * sizeof(double))];

    double temp_sum_delta = 0.0;
    double temp_sum_gamma = 0.0;

    for (size_t i = grid_rank; i < num_rows; i += grid_size) {
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

__global__ void addLocalDotContributions(double *dot_result_delta, double *dot_result_gamma) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        atomicAdd(dot_result_delta, grid_dot_result_delta);
        atomicAdd(dot_result_gamma, grid_dot_result_gamma);

        grid_dot_result_delta = 0.0;
        grid_dot_result_gamma = 0.0;
    }
}

__global__ void resetLocalDotProducts(double *dot_result_delta, double *dot_result_gamma) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        *dot_result_delta = 0.0;
        *dot_result_gamma = 0.0;
    }
}

double run_single_gpu(const int iter_max, int *um_I, int *um_J, real *um_val, real *x_ref,
                      int num_rows, int nnz) {
    real *um_x;
    real *um_r;
    real *um_p;
    real *um_s;
    real *um_z;
    real *um_w;
    real *um_q;
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

    cudaStream_t streamOtherOps;
    cudaStream_t streamSaxpy;
    cudaStream_t streamDot;
    cudaStream_t streamSpMV;

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
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_z, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_w, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_q, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_ax0, num_rows * sizeof(real)));

    CUDA_RT_CALL(cudaMallocManaged((void **)&um_alpha, sizeof(real)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_negative_alpha, sizeof(real)));
    CUDA_RT_CALL(cudaMallocManaged((void **)&um_beta, sizeof(real)));

    int sMemSize = 2 * (sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1));

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int numBlocks = (num_rows / THREADS_PER_BLOCK) + 1;

    CUDA_RT_CALL(cudaStreamCreate(&streamOtherOps));
    CUDA_RT_CALL(cudaStreamCreate(&streamDot));
    CUDA_RT_CALL(cudaStreamCreate(&streamSaxpy));
    CUDA_RT_CALL(cudaStreamCreate(&streamSpMV));

    CUDA_RT_CALL(cudaDeviceSynchronize());

    double start = omp_get_wtime();

    SingleGPU::initVectors<<<numBlocks, THREADS_PER_BLOCK, 0, streamOtherOps>>>(um_r, um_x,
                                                                                num_rows);

    CUDA_RT_CALL(cudaStreamSynchronize(streamOtherOps));

    // ax0 = Ax0
    SingleGPU::gpuSpMV<<<numBlocks, THREADS_PER_BLOCK, 0, streamOtherOps>>>(
        um_I, um_J, um_val, nnz, num_rows, real_positive_one, um_x, um_ax0);

    CUDA_RT_CALL(cudaStreamSynchronize(streamOtherOps));

    // r0 = b0 - s0
    // NOTE: b is a unit vector.
    SingleGPU::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, streamOtherOps>>>(
        um_ax0, um_r, real_negative_one, num_rows);

    CUDA_RT_CALL(cudaStreamSynchronize(streamOtherOps));

    // w0 = Ar0
    SingleGPU::gpuSpMV<<<numBlocks, THREADS_PER_BLOCK, 0, streamOtherOps>>>(
        um_I, um_J, um_val, nnz, num_rows, real_positive_one, um_r, um_w);

    CUDA_RT_CALL(cudaStreamSynchronize(streamOtherOps));

    int k = 1;

    while (k <= iter_max) {
        // Two dot products => <r, r> and <r, w>
        resetLocalDotProducts<<<1, 1, 0, streamDot>>>(um_tmp_dot_delta1, um_tmp_dot_gamma1);

        CUDA_RT_CALL(cudaStreamSynchronize(streamDot));

        gpuDotProductsMerged<<<numBlocks, THREADS_PER_BLOCK, sMemSize, streamDot>>>(
            um_r, um_r, um_r, um_w, num_rows, sMemSize);

        CUDA_RT_CALL(cudaStreamSynchronize(streamDot));

        addLocalDotContributions<<<1, 1, 0, streamDot>>>(um_tmp_dot_delta1, um_tmp_dot_gamma1);

        CUDA_RT_CALL(cudaStreamSynchronize(streamDot));

        // SpMV
        SingleGPU::gpuSpMV<<<numBlocks, THREADS_PER_BLOCK, 0, streamSpMV>>>(
            um_I, um_J, um_val, nnz, num_rows, real_positive_one, um_w, um_q);

        CUDA_RT_CALL(cudaStreamSynchronize(streamDot));
        CUDA_RT_CALL(cudaStreamSynchronize(streamSpMV));

        if (k > 1) {
            SingleGPU::update_b_k<<<1, 1, 0, streamOtherOps>>>((real)*um_tmp_dot_delta1,
                                                               *um_tmp_dot_delta0, um_beta);
            SingleGPU::update_a_k<<<1, 1, 0, streamOtherOps>>>(
                (real)*um_tmp_dot_delta1, (real)*um_tmp_dot_gamma1, *um_beta, um_alpha);
        } else {
            SingleGPU::init_b_k<<<1, 1, 0, streamOtherOps>>>(um_beta);
            SingleGPU::init_a_k<<<1, 1, 0, streamOtherOps>>>((real)*um_tmp_dot_delta1,
                                                             (real)*um_tmp_dot_gamma1, um_alpha);
        }

        CUDA_RT_CALL(cudaStreamSynchronize(streamOtherOps));

        // We don't need to sync streamSpMV until this point.

        // z_k = q_k + beta_k * z_(k-1)
        SingleGPU::gpuScaleVectorAndSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, streamSaxpy>>>(
            um_q, um_z, real_positive_one, *um_beta, num_rows);

        // s_k = w_k + beta_k * s_(k-1)
        SingleGPU::gpuScaleVectorAndSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, streamSaxpy>>>(
            um_w, um_s, real_positive_one, *um_beta, num_rows);

        // p_k = r_k = beta_k * p_(k-1)
        SingleGPU::gpuScaleVectorAndSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, streamSaxpy>>>(
            um_r, um_p, real_positive_one, *um_beta, num_rows);

        CUDA_RT_CALL(cudaStreamSynchronize(streamSaxpy));

        // x_(i+1) = x_i + alpha_i * p_i
        SingleGPU::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, streamSaxpy>>>(um_p, um_x, *um_alpha,
                                                                              num_rows);

        CUDA_RT_CALL(cudaStreamSynchronize(streamSaxpy));

        SingleGPU::a_minus<<<1, 1, 0, streamSaxpy>>>(*um_alpha, um_negative_alpha);

        CUDA_RT_CALL(cudaStreamSynchronize(streamSaxpy));

        // r_(i+1) = r_i - alpha_i * s_i
        SingleGPU::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, streamSaxpy>>>(
            um_s, um_r, *um_negative_alpha, num_rows);

        // w_(i+1) = w_i - alpha_i * z_i
        SingleGPU::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, streamSaxpy>>>(
            um_z, um_w, *um_negative_alpha, num_rows);

        *um_tmp_dot_delta0 = *um_tmp_dot_delta1;
        *um_tmp_dot_gamma0 = *um_tmp_dot_gamma1;

        CUDA_RT_CALL(cudaStreamSynchronize(streamSaxpy));

        k++;
    }

    double stop = omp_get_wtime();

    for (int i = 0; i < num_rows; i++) {
        x_ref[i] = um_x[i];
    }

    CUDA_RT_CALL(cudaFree(um_x));
    CUDA_RT_CALL(cudaFree(um_r));
    CUDA_RT_CALL(cudaFree(um_p));
    CUDA_RT_CALL(cudaFree(um_s));
    CUDA_RT_CALL(cudaFree(um_z));
    CUDA_RT_CALL(cudaFree(um_w));
    CUDA_RT_CALL(cudaFree(um_q));
    CUDA_RT_CALL(cudaFree(um_ax0));

    CUDA_RT_CALL(cudaFree(um_tmp_dot_delta1));
    CUDA_RT_CALL(cudaFree(um_tmp_dot_gamma1));
    CUDA_RT_CALL(cudaFree(um_tmp_dot_delta0));
    CUDA_RT_CALL(cudaFree(um_tmp_dot_gamma0));
    CUDA_RT_CALL(cudaFree(um_alpha));
    CUDA_RT_CALL(cudaFree(um_negative_alpha));
    CUDA_RT_CALL(cudaFree(um_beta));

    return (stop - start);
}
}  // namespace SingleGPUPipelinedDiscrete

namespace SingleGPUStandardDiscrete {
__device__ double grid_dot_result = 0.0;

__global__ void gpuDotProduct(real *vecA, real *vecB, int num_rows) {
    cg::thread_block cta = cg::this_thread_block();

    size_t grid_size = gridDim.x * blockDim.x;
    size_t grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double tmp[];

    double temp_sum = 0.0;

    for (size_t i = grid_rank; i < num_rows; i += grid_size) {
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
        atomicAdd(dot_result, grid_dot_result);

        grid_dot_result = 0.0;
    }
}

__global__ void resetLocalDotProduct(double *dot_result) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        *dot_result = 0.0;
    }
}

double run_single_gpu(const int iter_max, int *um_I, int *um_J, real *um_val, real *x_ref,
                      int num_rows, int nnz) {
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

    int sMemSize = (sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1));

    int numBlocks = (num_rows / THREADS_PER_BLOCK) + 1;

    CUDA_RT_CALL(cudaDeviceSynchronize());

    double start = omp_get_wtime();

    SingleGPU::initVectors<<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(um_r, um_x, num_rows);

    CUDA_RT_CALL(cudaStreamSynchronize(0));

    // ax0 = Ax0
    SingleGPU::gpuSpMV<<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(um_I, um_J, um_val, nnz, num_rows,
                                                               real_positive_one, um_x, um_ax0);

    CUDA_RT_CALL(cudaStreamSynchronize(0));

    // r0 = b0 - ax0
    // NOTE: b is a unit vector.
    SingleGPU::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(um_ax0, um_r, real_negative_one,
                                                                num_rows);

    CUDA_RT_CALL(cudaStreamSynchronize(0));

    // p0 = r0
    SingleGPU::gpuCopyVector<<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(um_r, um_p, num_rows);

    resetLocalDotProduct<<<1, 1, 0, 0>>>(um_tmp_dot_gamma1);

    CUDA_RT_CALL(cudaStreamSynchronize(0));

    gpuDotProduct<<<numBlocks, THREADS_PER_BLOCK, sMemSize, 0>>>(um_r, um_r, num_rows);

    CUDA_RT_CALL(cudaStreamSynchronize(0));

    addLocalDotContribution<<<1, 1, 0, 0>>>(um_tmp_dot_gamma1);

    CUDA_RT_CALL(cudaStreamSynchronize(0));

    *um_tmp_dot_gamma0 = *um_tmp_dot_gamma1;

    CUDA_RT_CALL(cudaStreamSynchronize(0));

    int k = 1;

    while (k <= iter_max) {
        // SpMV
        SingleGPU::gpuSpMV<<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(
            um_I, um_J, um_val, nnz, num_rows, real_positive_one, um_p, um_s);

        resetLocalDotProduct<<<1, 1, 0, 0>>>(um_tmp_dot_delta1);

        CUDA_RT_CALL(cudaStreamSynchronize(0));

        gpuDotProduct<<<numBlocks, THREADS_PER_BLOCK, sMemSize, 0>>>(um_p, um_s, num_rows);

        CUDA_RT_CALL(cudaStreamSynchronize(0));

        addLocalDotContribution<<<1, 1, 0, 0>>>(um_tmp_dot_delta1);

        CUDA_RT_CALL(cudaStreamSynchronize(0));

        SingleGPU::r1_div_x<<<1, 1, 0, 0>>>(*um_tmp_dot_gamma0, (real)*um_tmp_dot_delta1, um_alpha);

        CUDA_RT_CALL(cudaStreamSynchronize(0));

        // x_(k+1) = x_k + alpha_k * p_k
        SingleGPU::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(um_p, um_x, *um_alpha,
                                                                    num_rows);

        CUDA_RT_CALL(cudaStreamSynchronize(0));

        SingleGPU::a_minus<<<1, 1, 0, 0>>>(*um_alpha, um_negative_alpha);

        CUDA_RT_CALL(cudaStreamSynchronize(0));

        // r_(k+1) = r_k - alpha_k * s
        SingleGPU::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(um_s, um_r, *um_negative_alpha,
                                                                    num_rows);

        resetLocalDotProduct<<<1, 1, 0, 0>>>(um_tmp_dot_gamma1);

        CUDA_RT_CALL(cudaStreamSynchronize(0));

        gpuDotProduct<<<numBlocks, THREADS_PER_BLOCK, sMemSize, 0>>>(um_r, um_r, num_rows);

        CUDA_RT_CALL(cudaStreamSynchronize(0));

        addLocalDotContribution<<<1, 1, 0, 0>>>(um_tmp_dot_gamma1);

        CUDA_RT_CALL(cudaStreamSynchronize(0));

        SingleGPU::r1_div_x<<<1, 1, 0, 0>>>((real)*um_tmp_dot_gamma1, *um_tmp_dot_gamma0, um_beta);

        CUDA_RT_CALL(cudaStreamSynchronize(0));

        // p_(k+1) = r_(k+1) = beta_k * p_(k)
        SingleGPU::gpuScaleVectorAndSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(
            um_r, um_p, real_positive_one, *um_beta, num_rows);

        *um_tmp_dot_delta0 = *um_tmp_dot_delta1;
        *um_tmp_dot_gamma0 = *um_tmp_dot_gamma1;

        CUDA_RT_CALL(cudaStreamSynchronize(0));

        k++;
    }

    double stop = omp_get_wtime();

    for (int i = 0; i < num_rows; i++) {
        x_ref[i] = um_x[i];
    }

    CUDA_RT_CALL(cudaFree(um_x));
    CUDA_RT_CALL(cudaFree(um_r));
    CUDA_RT_CALL(cudaFree(um_p));
    CUDA_RT_CALL(cudaFree(um_s));
    CUDA_RT_CALL(cudaFree(um_ax0));

    CUDA_RT_CALL(cudaFree(um_tmp_dot_delta1));
    CUDA_RT_CALL(cudaFree(um_tmp_dot_gamma1));
    CUDA_RT_CALL(cudaFree(um_tmp_dot_delta0));
    CUDA_RT_CALL(cudaFree(um_tmp_dot_gamma0));
    CUDA_RT_CALL(cudaFree(um_alpha));
    CUDA_RT_CALL(cudaFree(um_negative_alpha));
    CUDA_RT_CALL(cudaFree(um_beta));

    return (stop - start);
}
}  // namespace SingleGPUStandardDiscrete

namespace CPU {
void cpuSpMV(int *I, int *J, real *val, int nnz, int num_rows, real alpha, real *inputVecX,
             real *outputVecY) {
    for (int i = 0; i < num_rows; i++) {
        int num_elems_this_row = I[i + 1] - I[i];

        real output = 0.0;
        for (int j = 0; j < num_elems_this_row; j++) {
            output += alpha * val[I[i] + j] * inputVecX[J[I[i] + j]];
        }
        outputVecY[i] = output;
    }

    return;
}

real dotProduct(real *vecA, real *vecB, int size) {
    real result = 0.0;

    for (int i = 0; i < size; i++) {
        result = result + (vecA[i] * vecB[i]);
    }

    return result;
}

void scaleVector(real *vec, real alpha, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = alpha * vec[i];
    }
}

void saxpy(real *x, real *y, real a, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = a * x[i] + y[i];
    }
}

void cpuConjugateGrad(const int iter_max, int *I, int *J, real *val, real *x, real *Ax, real *p,
                      real *r, int nnz, int num_rows, real tol) {
    int max_iter = iter_max;

    real alpha = 1.0;
    real alpham1 = -1.0;
    real r0 = 0.0;
    real b;
    real a;
    real na;

    cpuSpMV(I, J, val, nnz, num_rows, alpha, x, Ax);
    saxpy(Ax, r, alpham1, num_rows);

    real r1 = dotProduct(r, r, num_rows);

    int k = 1;

    while (k <= max_iter) {
        if (k > 1) {
            b = r1 / r0;
            scaleVector(p, b, num_rows);

            saxpy(r, p, alpha, num_rows);
        } else {
            for (int i = 0; i < num_rows; i++) p[i] = r[i];
        }

        cpuSpMV(I, J, val, nnz, num_rows, alpha, p, Ax);

        real dot = dotProduct(p, Ax, num_rows);
        a = r1 / dot;

        saxpy(p, x, a, num_rows);
        na = -a;
        saxpy(Ax, r, na, num_rows);

        r0 = r1;
        r1 = dotProduct(r, r, num_rows);

        k++;
    }
}
}  // namespace CPU