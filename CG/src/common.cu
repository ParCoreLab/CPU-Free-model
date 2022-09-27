#include "../include/common.h"

#include <omp.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

bool get_arg(char **begin, char **end, const std::string &arg) {
    char **itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nnz) {
    I[0] = 0, J[0] = 0, J[1] = 1;
    val[0] = (float)rand() / RAND_MAX + 10.0f;
    val[1] = (float)rand() / RAND_MAX;
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
        val[start + 1] = (float)rand() / RAND_MAX + 10.0f;

        if (i < N - 1) {
            val[start + 2] = (float)rand() / RAND_MAX;
        }
    }

    I[N] = nnz;
}

void report_results(const int num_rows, float *x_ref, float *x, const int num_devices,
                    const double single_gpu_runtime, const double start, const double stop,
                    const bool compare_to_single_gpu) {
    bool result_correct = true;

    if (compare_to_single_gpu) {
        for (int i = 0; i < num_rows; i++) {
            if (std::fabs(x_ref[i] - x[i]) > tol) {
                fprintf(stderr,
                        "ERROR: x[%d] = %.8f does not match %.8f "
                        "(reference)\n",
                        i, x[i], x_ref[i]);
                // result_correct = false;
            }
        }
    }

    if (result_correct) {
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

// Single GPU Pipelined Implementation

namespace SingleGPUPipelinedNonPersistent {
__device__ double grid_dot_result_delta = 0.0;
__device__ double grid_dot_result_gamma = 0.0;

__global__ void initVectors(float *r, float *x, int num_rows) {
    size_t grid_size = gridDim.x * blockDim.x;
    size_t grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = grid_rank; i < num_rows; i += grid_size) {
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
                        float *inputVecX, float *outputVecY) {
    size_t grid_size = gridDim.x * blockDim.x;
    size_t grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = grid_rank; i < num_rows; i += grid_size) {
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

__global__ void gpuSaxpy(float *x, float *y, float *a, int num_rows) {
    size_t grid_size = gridDim.x * blockDim.x;
    size_t grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = grid_rank; i < num_rows; i += grid_size) {
        y[i] = a[0] * x[i] + y[i];
    }
}

// Performs two dot products at the same time
// Used to perform <r, r> and <r, w> at the same time
// Can we combined the two atomicAdds somehow?

__global__ void gpuDotProductsMerged(float *vecA_delta, float *vecB_delta, float *vecA_gamma,
                                     float *vecB_gamma, int num_rows, const int sMemSize) {
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

__global__ void gpuCopyVector(float *srcA, float *destB, int num_rows) {
    size_t grid_size = gridDim.x * blockDim.x;
    size_t grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = grid_rank; i < num_rows; i += grid_size) {
        destB[i] = srcA[i];
    }
}

__global__ void gpuScaleVectorAndSaxpy(float *x, float *y, float a, float *scale, int num_rows) {
    size_t grid_size = gridDim.x * blockDim.x;
    size_t grid_rank = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = grid_rank; i < num_rows; i += grid_size) {
        y[i] = a * x[i] + scale[0] * y[i];
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

double run_single_gpu(const int iter_max, char *matrix_path_char,
                      bool generate_random_tridiag_matrix, int *um_I, int *um_J, float *um_val,
                      float *const x_ref, int num_rows, int nnz) {
    CUDA_RT_CALL(cudaSetDevice(0));

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

    float rhs = 1.0;
    float r1;
    float float_positive_one = 1.0;
    float float_negative_one = -1.0;

    cudaStream_t streamOtherOps;
    cudaStream_t streamSaxpy;
    cudaStream_t streamDot;
    cudaStream_t streamSpMV;

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

    int sMemSize = 2 * (sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1));

    CUDA_RT_CALL(cudaSetDevice(0));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int numSms = deviceProp.multiProcessorCount;

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

    CUDA_RT_CALL(cudaStreamCreate(&streamOtherOps));
    CUDA_RT_CALL(cudaStreamCreate(&streamDot));
    CUDA_RT_CALL(cudaStreamCreate(&streamSaxpy));
    CUDA_RT_CALL(cudaStreamCreate(&streamSpMV));

    CUDA_RT_CALL(cudaDeviceSynchronize());

    double start = omp_get_wtime();

    initVectors<<<initVectorsGridSize, THREADS_PER_BLOCK, 0, streamOtherOps>>>(um_r, um_x,
                                                                               num_rows);

    CUDA_RT_CALL(cudaDeviceSynchronize());

    gpuSpMV<<<spmvGridSize, THREADS_PER_BLOCK, 0, streamOtherOps>>>(
        um_I, um_J, um_val, nnz, num_rows, float_positive_one, um_x, um_s);

    CUDA_RT_CALL(cudaDeviceSynchronize());

    gpuSaxpy<<<saxpyGridSize, THREADS_PER_BLOCK, 0, streamOtherOps>>>(
        um_s, um_r, um_float_negative_one, num_rows);

    CUDA_RT_CALL(cudaDeviceSynchronize());

    // w0 = Ar0
    gpuSpMV<<<spmvGridSize, THREADS_PER_BLOCK, 0, streamOtherOps>>>(
        um_I, um_J, um_val, nnz, num_rows, float_positive_one, um_r, um_w);

    CUDA_RT_CALL(cudaDeviceSynchronize());

    gpuDotProductsMerged<<<dotProductGridSize, THREADS_PER_BLOCK, sMemSize, streamOtherOps>>>(
        um_r, um_r, um_r, um_w, num_rows, sMemSize);

    CUDA_RT_CALL(cudaDeviceSynchronize());

    addLocalDotContributions<<<1, 1, 0, streamOtherOps>>>(um_dot_result_delta, um_dot_result_gamma);

    CUDA_RT_CALL(cudaDeviceSynchronize());

    um_tmp_dot_delta1[0] = um_dot_result_delta[0];

    int k = 1;

    while (k <= iter_max) {
        if (k > 1) {
            update_b_k<<<1, 1, 0, streamOtherOps>>>(um_tmp_dot_delta1, um_tmp_dot_delta0, um_b);

            gpuScaleVectorAndSaxpy<<<scaleVectorAndSaxpyGridSize, THREADS_PER_BLOCK, 0,
                                     streamOtherOps>>>(um_r, um_p, float_positive_one, um_b,
                                                       num_rows);

            gpuScaleVectorAndSaxpy<<<scaleVectorAndSaxpyGridSize, THREADS_PER_BLOCK, 0,
                                     streamOtherOps>>>(um_w, um_s, float_positive_one, um_b,
                                                       num_rows);

            gpuScaleVectorAndSaxpy<<<scaleVectorAndSaxpyGridSize, THREADS_PER_BLOCK, 0,
                                     streamOtherOps>>>(um_t, um_u, float_positive_one, um_b,
                                                       num_rows);
        } else {
            gpuCopyVector<<<copyVectorGridSize, THREADS_PER_BLOCK, 0, streamOtherOps>>>(um_r, um_p,
                                                                                        num_rows);

            // Need to figure out what to copy where
            // Other vectors also need to be initialized
            // Fine for now

            // gpuCopyVector<<<copyVectorGridSize, THREADS_PER_BLOCK, 0, nStreams[gpu_idx]>>>(
            //     um_s, um_w, num_rows, gpu_idx, num_devices);
        }

        CUDA_RT_CALL(cudaDeviceSynchronize());

        // SpMV
        gpuSpMV<<<spmvGridSize, THREADS_PER_BLOCK, sMemSize, streamSpMV>>>(
            um_I, um_J, um_val, nnz, num_rows, float_positive_one, um_p, um_s);

        // Two dot products => <r, r> and <r, w>
        resetLocalDotProducts<<<1, 1, 0, streamDot>>>(um_dot_result_delta, um_dot_result_gamma);

        gpuDotProductsMerged<<<dotProductGridSize, THREADS_PER_BLOCK, sMemSize, streamDot>>>(
            um_r, um_r, um_r, um_w, num_rows, sMemSize);

        addLocalDotContributions<<<1, 1, 0, streamDot>>>(um_dot_result_delta, um_dot_result_gamma);

        CUDA_RT_CALL(cudaDeviceSynchronize());

        update_a_k<<<1, 1, 0, streamSaxpy>>>((float *)um_dot_result_delta,
                                             (float *)um_dot_result_gamma, um_b, um_a);

        gpuSaxpy<<<saxpyGridSize, THREADS_PER_BLOCK, 0, streamSaxpy>>>(um_p, um_x, um_a, num_rows);

        a_minus<<<1, 1, 0, streamSaxpy>>>(um_a, um_na);

        gpuSaxpy<<<saxpyGridSize, THREADS_PER_BLOCK, 0, streamSaxpy>>>(um_s, um_r, um_na, num_rows);

        gpuSaxpy<<<saxpyGridSize, THREADS_PER_BLOCK, 0, streamSaxpy>>>(um_u, um_w, um_na, num_rows);

        um_tmp_dot_delta0[0] = um_tmp_dot_delta1[0];
        um_tmp_dot_gamma0[0] = um_tmp_dot_gamma1[0];

        um_tmp_dot_delta1[0] = um_dot_result_delta[0];
        um_tmp_dot_gamma1[0] = um_dot_result_gamma[0];

        CUDA_RT_CALL(cudaStreamSynchronize(0));

        k++;
    }

    r1 = (float)um_dot_result_gamma[0];

    double stop = omp_get_wtime();

    for (int i = 0; i < num_rows; i++) {
        x_ref[i] = um_x[i];
    }

    return (stop - start);
}
}  // namespace SingleGPUPipelinedNonPersistent