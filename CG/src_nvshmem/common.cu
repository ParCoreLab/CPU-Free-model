#include "../include_nvshmem/common.h"

#include <math.h>
#include <omp.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#include <iostream>

namespace cg = cooperative_groups;

bool get_arg(char **begin, char **end, const std::string &arg) {
    char **itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

long long unsigned int parse_nvshmem_symmetric_size(char *value) {
    long long unsigned int units, size;

    assert(value != NULL);

    if (strchr(value, 'G') != NULL) {
        units = 1e9;
    } else if (strchr(value, 'M') != NULL) {
        units = 1e6;
    } else if (strchr(value, 'K') != NULL) {
        units = 1e3;
    } else {
        units = 1;
    }

    assert(atof(value) >= 0);
    size = (long long unsigned int)atof(value) * units;

    return size;
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

void report_errors(const int num_rows, real *x_ref_single_gpu, real *x_ref_cpu, real *x,
                   int row_start_idx, int row_end_idx, const int num_devices,
                   const double single_gpu_runtime, const double start, const double stop,
                   const bool compare_to_single_gpu, const bool compare_to_cpu,
                   bool &result_correct_single_gpu, bool &result_correct_cpu) {
    result_correct_single_gpu = true;
    result_correct_cpu = true;

    int i = 0;

    if (compare_to_single_gpu) {
        for (i = row_start_idx; result_correct_single_gpu && (i < row_end_idx); i++) {
            if (std::fabs(x_ref_single_gpu[i] - x[i]) > tol || isnan(x[i]) ||
                isnan(x_ref_single_gpu[i])) {
                fprintf(stderr,
                        "ERROR: x[%d] = %.8f does not match %.8f "
                        "(Single GPU reference)\n",
                        i, x[i], x_ref_single_gpu[i]);

                result_correct_single_gpu = false;
            }
        }
    }

    if (compare_to_cpu) {
        for (i = row_start_idx; result_correct_cpu && (i < row_end_idx); i++) {
            if (std::fabs(x_ref_cpu[i] - x[i]) > tol || isnan(x[i]) || isnan(x_ref_cpu[i])) {
                fprintf(stderr,
                        "ERROR: x[%d] = %.8f does not match %.8f "
                        "(CPU reference)\n",
                        i, x[i], x_ref_cpu[i]);

                result_correct_cpu = false;
            }
        }
    }
}

void report_runtime(const int num_devices, const double single_gpu_runtime, const double start,
                    const double stop, const bool result_correct_single_gpu,
                    const bool result_correct_cpu, const bool compare_to_single_gpu) {
    if (result_correct_single_gpu && result_correct_cpu) {
        printf("Execution time: %8.4f s\n", (stop - start));

        if (result_correct_single_gpu && result_correct_cpu && compare_to_single_gpu) {
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

__global__ void gpuSpMV(int *rowInd, int *colInd, real *val, int nnz, int num_rows, real alpha,
                        real *inputVecX, real *outputVecY, bool matrix_is_zero_indexed) {
    size_t grid_size = gridDim.x * blockDim.x;
    size_t grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = grid_rank; i < num_rows; i += grid_size) {
        int row_elem = rowInd[i] - int(!matrix_is_zero_indexed);
        int next_row_elem = rowInd[i + 1] - int(!matrix_is_zero_indexed);
        int num_elems_this_row = next_row_elem - row_elem;

        real output = 0.0;
        for (int j = 0; j < num_elems_this_row; j++) {
            int input_vec_elem_idx = colInd[row_elem + j] - int(!matrix_is_zero_indexed);

            output += alpha * val[row_elem + j] * inputVecX[input_vec_elem_idx];
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

// Common kernels used by NVSHMEM versions
namespace NVSHMEM {
__global__ void initVectors(real *r, real *x, int row_start_idx, int chunk_size, int num_rows) {
    int grid_rank = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;

    for (int local_row_idx = grid_rank; local_row_idx < chunk_size; local_row_idx += grid_size) {
        int global_row_idx = row_start_idx + local_row_idx;

        if (global_row_idx < num_rows) {
            r[local_row_idx] = 1.0;
            x[local_row_idx] = 0.0;
        }
    }
}

__global__ void gpuSpMV(int *rowInd, int *colInd, real *val, real alpha, real *inputVecX,
                        real *outputVecY, int row_start_idx, int chunk_size, int num_rows,
                        bool matrix_is_zero_indexed) {
    int grid_rank = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;

    int mype = nvshmem_my_pe();

    for (int local_row_idx = grid_rank; local_row_idx < chunk_size; local_row_idx += grid_size) {
        int global_row_idx = row_start_idx + local_row_idx;

        if (global_row_idx < num_rows) {
            int row_elem = rowInd[global_row_idx] - int(!matrix_is_zero_indexed);
            int next_row_elem = rowInd[global_row_idx + 1] - int(!matrix_is_zero_indexed);
            int num_elems_this_row = next_row_elem - row_elem;

            real output = 0.0;

            for (int j = 0; j < num_elems_this_row; j++) {
                int input_vec_elem_idx = colInd[row_elem + j] - int(!matrix_is_zero_indexed);
                int remote_pe = input_vec_elem_idx / chunk_size;

                int remote_pe_idx_offset = input_vec_elem_idx - remote_pe * chunk_size;

                real elem_val = nvshmem_double_g(inputVecX + remote_pe_idx_offset, remote_pe);

                output += alpha * val[row_elem + j] * elem_val;
            }

            outputVecY[local_row_idx] = output;
        }
    }
}

__global__ void gpuSaxpy(real *x, real *y, real a, int chunk_size) {
    size_t grid_rank = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * blockDim.x;

    for (size_t i = grid_rank; i < chunk_size; i += grid_size) {
        y[i] = a * x[i] + y[i];
    }
}

__global__ void gpuCopyVector(real *srcA, real *destB, int chunk_size) {
    size_t grid_rank = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * blockDim.x;

    for (size_t i = grid_rank; i < chunk_size; i += grid_size) {
        destB[i] = srcA[i];
    }
}

__global__ void gpuScaleVectorAndSaxpy(real *x, real *y, real a, real scale, int chunk_size) {
    size_t grid_rank = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * blockDim.x;

    for (size_t i = grid_rank; i < chunk_size; i += grid_size) {
        y[i] = a * x[i] + scale * y[i];
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
}  // namespace NVSHMEM

namespace SingleGPUDiscreteStandard {
__global__ void gpuDotProduct(real *vecA, real *vecB, double *local_dot_result, int num_rows) {
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
            atomicAdd(local_dot_result, temp_sum);
        }
    }
}

__global__ void resetLocalDotProduct(double *dot_result) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        *dot_result = 0.0;
    }
}

double run_single_gpu(const int iter_max, int *device_csrRowIndices, int *device_csrColIndices,
                      real *device_csrVal, real *x_ref, int num_rows, int nnz,
                      bool matrix_is_zero_indexed, bool run_as_separate_version) {
    real *device_x;
    real *device_r;
    real *device_p;
    real *device_s;
    real *device_ax0;

    real tmp_dot_gamma0;

    real alpha;
    real negative_alpha;
    real beta;

    double *device_dot_delta1;
    double *device_dot_gamma1;
    double host_dot_gamma1;
    double host_dot_delta1;

    real real_positive_one = 1.0;
    real real_negative_one = -1.0;

    CUDA_RT_CALL(cudaMalloc((void **)&device_x, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc((void **)&device_r, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc((void **)&device_p, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc((void **)&device_s, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc((void **)&device_ax0, num_rows * sizeof(real)));

    CUDA_RT_CALL(cudaMalloc((void **)&device_dot_delta1, sizeof(double)));
    CUDA_RT_CALL(cudaMalloc((void **)&device_dot_gamma1, sizeof(double)));

    CUDA_RT_CALL(cudaMemset(device_dot_delta1, 0, sizeof(double)));
    CUDA_RT_CALL(cudaMemset(device_dot_gamma1, 0, sizeof(double)));

    int sMemSize = (sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1));

    int numBlocks = (num_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    double start = omp_get_wtime();

    SingleGPU::initVectors<<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(device_r, device_x, num_rows);

    // ax0 = Ax0
    SingleGPU::gpuSpMV<<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(
        device_csrRowIndices, device_csrColIndices, device_csrVal, nnz, num_rows, real_positive_one,
        device_x, device_ax0, matrix_is_zero_indexed);

    // r0 = b0 - ax0
    // NOTE: b is a unit vector.
    SingleGPU::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(device_ax0, device_r,
                                                                real_negative_one, num_rows);

    // p0 = r0
    SingleGPU::gpuCopyVector<<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(device_r, device_p, num_rows);

    resetLocalDotProduct<<<1, 1, 0, 0>>>(device_dot_gamma1);

    gpuDotProduct<<<numBlocks, THREADS_PER_BLOCK, sMemSize, 0>>>(device_r, device_r,
                                                                 device_dot_gamma1, num_rows);

    CUDA_RT_CALL(cudaMemcpyAsync(&host_dot_gamma1, device_dot_gamma1, sizeof(double),
                                 cudaMemcpyDeviceToHost, 0));

    CUDA_RT_CALL(cudaDeviceSynchronize());

    tmp_dot_gamma0 = host_dot_gamma1;

    int k = 0;

    while (k < iter_max) {
        // SpMV
        SingleGPU::gpuSpMV<<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(
            device_csrRowIndices, device_csrColIndices, device_csrVal, nnz, num_rows,
            real_positive_one, device_p, device_s, matrix_is_zero_indexed);

        resetLocalDotProduct<<<1, 1, 0, 0>>>(device_dot_delta1);

        gpuDotProduct<<<numBlocks, THREADS_PER_BLOCK, sMemSize, 0>>>(device_p, device_s,
                                                                     device_dot_delta1, num_rows);

        CUDA_RT_CALL(cudaMemcpyAsync(&host_dot_delta1, device_dot_delta1, sizeof(double),
                                     cudaMemcpyDeviceToHost, 0));

        CUDA_RT_CALL(cudaDeviceSynchronize());

        alpha = tmp_dot_gamma0 / (real)host_dot_delta1;

        // x_(k+1) = x_k + alpha_k * p_k
        SingleGPU::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(device_p, device_x, alpha,
                                                                    num_rows);

        negative_alpha = -alpha;

        // r_(k+1) = r_k - alpha_k * s
        SingleGPU::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(device_s, device_r,
                                                                    negative_alpha, num_rows);

        resetLocalDotProduct<<<1, 1, 0, 0>>>(device_dot_gamma1);

        gpuDotProduct<<<numBlocks, THREADS_PER_BLOCK, sMemSize, 0>>>(device_r, device_r,
                                                                     device_dot_gamma1, num_rows);

        CUDA_RT_CALL(cudaMemcpyAsync(&host_dot_gamma1, device_dot_gamma1, sizeof(double),
                                     cudaMemcpyDeviceToHost, 0));

        CUDA_RT_CALL(cudaDeviceSynchronize());

        beta = (real)host_dot_gamma1 / tmp_dot_gamma0;

        // p_(k+1) = r_(k+1) = beta_k * p_(k)
        SingleGPU::gpuScaleVectorAndSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, 0>>>(
            device_r, device_p, real_positive_one, beta, num_rows);

        tmp_dot_gamma0 = host_dot_gamma1;

        CUDA_RT_CALL(cudaDeviceSynchronize());

        k++;
    }

    double stop = omp_get_wtime();

    if (!run_as_separate_version) {
        CUDA_RT_CALL(cudaMemcpy(x_ref, device_x, num_rows * sizeof(real), cudaMemcpyDeviceToHost));
    }

    CUDA_RT_CALL(cudaFree(device_x));
    CUDA_RT_CALL(cudaFree(device_r));
    CUDA_RT_CALL(cudaFree(device_p));
    CUDA_RT_CALL(cudaFree(device_s));
    CUDA_RT_CALL(cudaFree(device_ax0));

    CUDA_RT_CALL(cudaFree(device_dot_delta1));
    CUDA_RT_CALL(cudaFree(device_dot_gamma1));

    return (stop - start);
}
}  // namespace SingleGPUDiscreteStandard

// Single GPU Pipelined Implementation

namespace SingleGPUDiscretePipelined {
__device__ double grid_dot_result_delta = 0.0;
__device__ double grid_dot_result_gamma = 0.0;

// Performs two dot products at the same time
// Used to perform <r, r> and <r, w> at the same time
// Can we combined the two atomicAdds somehow?

__global__ void gpuDotProductsMerged(real *vecA_delta, real *vecB_delta, real *vecA_gamma,
                                     real *vecB_gamma, double *local_dot_result_delta,
                                     double *local_dot_result_gamma, int num_rows,
                                     const int sMemSize) {
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
            atomicAdd(local_dot_result_delta, temp_sum_delta);
            atomicAdd(local_dot_result_gamma, temp_sum_gamma);
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

double run_single_gpu(const int iter_max, int *device_csrRowIndices, int *device_csrColIndices,
                      real *device_csrVal, real *x_ref, int num_rows, int nnz,
                      bool matrix_is_zero_indexed) {
    real *device_x;
    real *device_r;
    real *device_p;
    real *device_s;
    real *device_z;
    real *device_w;
    real *device_q;
    real *device_ax0;

    real alpha;
    real negative_alpha;
    real beta;

    real tmp_dot_delta0;

    double *device_dot_delta1;
    double *device_dot_gamma1;
    double host_dot_gamma1;
    double host_dot_delta1;

    real real_positive_one = 1.0;
    real real_negative_one = -1.0;

    cudaStream_t streamOtherOps;
    cudaStream_t streamSaxpy;
    cudaStream_t streamDot;
    cudaStream_t streamSpMV;

    CUDA_RT_CALL(cudaMalloc((void **)&device_x, sizeof(real) * num_rows));

    CUDA_RT_CALL(cudaMalloc((void **)&device_dot_delta1, sizeof(double)));
    CUDA_RT_CALL(cudaMalloc((void **)&device_dot_gamma1, sizeof(double)));

    CUDA_RT_CALL(cudaMemset(device_dot_delta1, 0, sizeof(double)));
    CUDA_RT_CALL(cudaMemset(device_dot_gamma1, 0, sizeof(double)));

    // temp memory for ConjugateGradient
    CUDA_RT_CALL(cudaMalloc((void **)&device_r, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc((void **)&device_p, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc((void **)&device_s, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc((void **)&device_z, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc((void **)&device_w, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc((void **)&device_q, num_rows * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc((void **)&device_ax0, num_rows * sizeof(real)));

    int sMemSize = 2 * (sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1));

    int numBlocks = (num_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    CUDA_RT_CALL(cudaStreamCreate(&streamOtherOps));
    CUDA_RT_CALL(cudaStreamCreate(&streamDot));
    CUDA_RT_CALL(cudaStreamCreate(&streamSaxpy));
    CUDA_RT_CALL(cudaStreamCreate(&streamSpMV));

    CUDA_RT_CALL(cudaDeviceSynchronize());

    double start = omp_get_wtime();

    SingleGPU::initVectors<<<numBlocks, THREADS_PER_BLOCK, 0, streamOtherOps>>>(device_r, device_x,
                                                                                num_rows);

    // ax0 = Ax0
    SingleGPU::gpuSpMV<<<numBlocks, THREADS_PER_BLOCK, 0, streamOtherOps>>>(
        device_csrRowIndices, device_csrColIndices, device_csrVal, nnz, num_rows, real_positive_one,
        device_x, device_ax0, matrix_is_zero_indexed);

    // r0 = b0 - s0
    // NOTE: b is a unit vector.
    SingleGPU::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, streamOtherOps>>>(
        device_ax0, device_r, real_negative_one, num_rows);

    // w0 = Ar0
    SingleGPU::gpuSpMV<<<numBlocks, THREADS_PER_BLOCK, 0, streamOtherOps>>>(
        device_csrRowIndices, device_csrColIndices, device_csrVal, nnz, num_rows, real_positive_one,
        device_r, device_w, matrix_is_zero_indexed);

    CUDA_RT_CALL(cudaStreamSynchronize(streamOtherOps));

    int k = 0;

    while (k < iter_max) {
        // Two dot products => <r, r> and <r, w>
        resetLocalDotProducts<<<1, 1, 0, streamDot>>>(device_dot_delta1, device_dot_gamma1);

        gpuDotProductsMerged<<<numBlocks, THREADS_PER_BLOCK, sMemSize, streamDot>>>(
            device_r, device_r, device_r, device_w, device_dot_delta1, device_dot_gamma1, num_rows,
            sMemSize);

        CUDA_RT_CALL(cudaMemcpyAsync(&host_dot_delta1, device_dot_delta1, sizeof(double),
                                     cudaMemcpyDeviceToHost, streamDot));

        CUDA_RT_CALL(cudaMemcpyAsync(&host_dot_gamma1, device_dot_gamma1, sizeof(double),
                                     cudaMemcpyDeviceToHost, streamDot));

        // SpMV
        SingleGPU::gpuSpMV<<<numBlocks, THREADS_PER_BLOCK, 0, streamSpMV>>>(
            device_csrRowIndices, device_csrColIndices, device_csrVal, nnz, num_rows,
            real_positive_one, device_w, device_q, matrix_is_zero_indexed);

        CUDA_RT_CALL(cudaStreamSynchronize(streamDot));

        real real_tmp_dot_delta1 = host_dot_delta1;
        real real_tmp_dot_gamma1 = host_dot_gamma1;

        if (k > 1) {
            beta = real_tmp_dot_delta1 / tmp_dot_delta0;
            alpha =
                real_tmp_dot_delta1 / (real_tmp_dot_gamma1 - (beta / alpha) * real_tmp_dot_delta1);
        } else {
            beta = 0.0;
            alpha = real_tmp_dot_delta1 / real_tmp_dot_gamma1;
        }

        CUDA_RT_CALL(cudaStreamSynchronize(streamSpMV));

        // z_k = q_k + beta_k * z_(k-1)
        SingleGPU::gpuScaleVectorAndSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, streamSaxpy>>>(
            device_q, device_z, real_positive_one, beta, num_rows);

        // s_k = w_k + beta_k * s_(k-1)
        SingleGPU::gpuScaleVectorAndSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, streamSaxpy>>>(
            device_w, device_s, real_positive_one, beta, num_rows);

        // p_k = r_k = beta_k * p_(k-1)
        SingleGPU::gpuScaleVectorAndSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, streamSaxpy>>>(
            device_r, device_p, real_positive_one, beta, num_rows);

        // x_(i+1) = x_i + alpha_i * p_i
        SingleGPU::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, streamSaxpy>>>(device_p, device_x,
                                                                              alpha, num_rows);

        negative_alpha = -alpha;

        // r_(i+1) = r_i - alpha_i * s_i
        SingleGPU::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, streamSaxpy>>>(
            device_s, device_r, negative_alpha, num_rows);

        // w_(i+1) = w_i - alpha_i * z_i
        SingleGPU::gpuSaxpy<<<numBlocks, THREADS_PER_BLOCK, 0, streamSaxpy>>>(
            device_z, device_w, negative_alpha, num_rows);

        tmp_dot_delta0 = host_dot_delta1;

        CUDA_RT_CALL(cudaStreamSynchronize(streamSaxpy));

        k++;
    }

    double stop = omp_get_wtime();

    CUDA_RT_CALL(cudaFree(device_x));
    CUDA_RT_CALL(cudaFree(device_r));
    CUDA_RT_CALL(cudaFree(device_p));
    CUDA_RT_CALL(cudaFree(device_s));
    CUDA_RT_CALL(cudaFree(device_z));
    CUDA_RT_CALL(cudaFree(device_w));
    CUDA_RT_CALL(cudaFree(device_q));
    CUDA_RT_CALL(cudaFree(device_ax0));

    CUDA_RT_CALL(cudaFree(device_dot_delta1));
    CUDA_RT_CALL(cudaFree(device_dot_gamma1));

    CUDA_RT_CALL(cudaStreamDestroy(streamOtherOps));
    CUDA_RT_CALL(cudaStreamDestroy(streamDot));
    CUDA_RT_CALL(cudaStreamDestroy(streamSaxpy));
    CUDA_RT_CALL(cudaStreamDestroy(streamSpMV));

    return (stop - start);
}
}  // namespace SingleGPUDiscretePipelined

namespace CPU {
void cpuSpMV(int *rowInd, int *colInd, real *val, int nnz, int num_rows, real alpha,
             real *inputVecX, real *outputVecY, bool matrix_is_zero_indexed) {
    for (int i = 0; i < num_rows; i++) {
        int row_elem = rowInd[i] - int(!matrix_is_zero_indexed);
        int next_row_elem = rowInd[i + 1] - int(!matrix_is_zero_indexed);
        int num_elems_this_row = next_row_elem - row_elem;

        real output = 0.0;
        for (int j = 0; j < num_elems_this_row; j++) {
            int input_vec_elem_idx = colInd[row_elem + j] - int(!matrix_is_zero_indexed);

            output += alpha * val[row_elem + j] * inputVecX[input_vec_elem_idx];
        }
    }
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

void cpuConjugateGrad(const int iter_max, int *host_csrRowIndices, int *host_csrColIndices,
                      real *host_csrVal, real *x, real *Ax, real *p, real *r, int nnz, int num_rows,
                      real tol, bool matrix_is_zero_indexed) {
    int max_iter = iter_max;

    real alpha = 1.0;
    real alpham1 = -1.0;
    real r0 = 0.0;
    real b;
    real a;
    real na;

    cpuSpMV(host_csrRowIndices, host_csrColIndices, host_csrVal, nnz, num_rows, alpha, x, Ax,
            matrix_is_zero_indexed);
    saxpy(Ax, r, alpham1, num_rows);

    real r1 = dotProduct(r, r, num_rows);

    int k = 0;

    while (k < max_iter) {
        if (k > 1) {
            b = r1 / r0;
            scaleVector(p, b, num_rows);

            saxpy(r, p, alpha, num_rows);
        } else {
            for (int i = 0; i < num_rows; i++) p[i] = r[i];
        }

        cpuSpMV(host_csrRowIndices, host_csrColIndices, host_csrVal, nnz, num_rows, alpha, p, Ax,
                matrix_is_zero_indexed);

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