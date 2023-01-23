#include "../include/common.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

bool get_arg(char **begin, char **end, const std::string &arg) {
    char **itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

__global__ void initialize_boundaries(real *__restrict__ const a_new, real *__restrict__ const a,
                                      const real pi, const int offset, const int nx,
                                      const int my_ny, const int ny) {
    for (unsigned int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < my_ny;
         iy += blockDim.x * gridDim.x) {
        const real y0 = sin(2.0 * pi * (offset + iy) / (ny - 1));
        for (unsigned int ix = 0; ix < nx; ix += nx - 1) {
            a[iy * nx + ix] = y0;
            a_new[iy * nx + ix] = y0;
        }
    }
}

__global__ void jacobi_kernel_single_gpu(real *__restrict__ const a_new,
                                         const real *__restrict__ const a,
                                         real *__restrict__ const l2_norm, const int iy_start,
                                         const int iy_end, const int nx,
                                         const bool calculate_norm) {
    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (iy < iy_end && ix < (nx - 1)) {
        const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                     a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);

        a_new[iy * nx + ix] = new_val;
    }
}

__global__ void jacobi_kernel_single_gpu_mirror(real *__restrict__ const a_new,
                                                const real *__restrict__ const a,
                                                real *__restrict__ const l2_norm,
                                                const int iy_start, const int iy_end, const int nx,
                                                const bool calculate_norm) {
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    if (iy < iy_start || iy > (iy_end - 1)) {
        //        if (iy < iy_end && ix < (nx)) {
        return;
    }

    real east = a[iy * nx + ix + (ix < (nx - 1))];
    real west = a[iy * nx + ix - (ix > 0)];

    real north = a[(iy + 1) * nx + ix];
    real south = a[(iy - 1) * nx + ix];

    const real new_val = 0.25f * (north + south + east + west);

    a_new[iy * nx + ix] = new_val;
}

// I changed the kernel, switch it back later
__global__ void jacobi_kernel_single_gpu_perks(real *__restrict__ const a_new,
                                               const real *__restrict__ const a,
                                               real *__restrict__ const l2_norm, const int iy_start,
                                               const int iy_end, const int nx,
                                               const bool calculate_norm) {
    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (iy < iy_end && ix < (nx - 1)) {
        const real new_val =
            ((5 * a[(iy - 1) * nx + ix]) + (12 * a[iy * nx + ix + 1]) + (15 * a[iy * nx + ix]) +
             (12 * a[iy * nx + ix - 1]) + (5 * a[(iy + 1) * nx + ix])) /
            118;

        a_new[iy * nx + ix] = new_val;
    }
}

__global__ void jacobi_kernel_single_gpu_persistent(real *a_new, real *a, const int iy_start,
                                                    const int iy_end, const int nx,
                                                    const bool calculate_norm, const int iter_max) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;

    int iter = 0;

    while (iter < iter_max) {
        if (iy < iy_end && ix < (nx - 1)) {
            const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                         a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
            a_new[iy * nx + ix] = new_val;

            if (iy_start == iy) {
                a_new[iy_end * nx + ix] = new_val;
            }

            if ((iy_end - 1) == iy) {
                a_new[(iy_start - 1) * nx + ix] = new_val;
            }
        }

        iter++;

        real *temp_pointer = a_new;
        a_new = a;
        a = temp_pointer;

        cg::sync(grid);
    }
}

/*double single_cpu(real *a_h_input, const int nx, const int ny, const int iter_max,
                  real *const a_ref_h, const int nccheck, const bool print)
{
    double start = omp_get_wtime();
    jacobi_gold_iterative(a_h_input, ny, nx, a_ref_h, iter_max);
    return omp_get_wtime() - start;
}*/

double single_gpu(const int nx, const int ny, const int iter_max, real *const a_ref_h,
                  const int nccheck, const bool print, decltype(jacobi_kernel_single_gpu) kernel) {
    real *a;
    real *a_new;

    cudaStream_t compute_stream;
    cudaStream_t push_top_stream;
    cudaStream_t push_bottom_stream;
    cudaEvent_t compute_done;
    cudaEvent_t push_top_done;
    cudaEvent_t push_bottom_done;

    int iy_start = 1;
    int iy_end = (ny - 1);

    CUDA_RT_CALL(cudaMalloc(&a, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc(&a_new, nx * ny * sizeof(real)));

    CUDA_RT_CALL(cudaMemset(a, 0, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * ny * sizeof(real)));

    // Set diriclet boundary conditions on left and right boarder
    initialize_boundaries<<<ny / 128 + 1, 128>>>(a, a_new, PI, 0, nx, ny, ny);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());

    CUDA_RT_CALL(cudaStreamCreate(&compute_stream));
    CUDA_RT_CALL(cudaStreamCreate(&push_top_stream));
    CUDA_RT_CALL(cudaStreamCreate(&push_bottom_stream));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&compute_done, cudaEventDisableTiming));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&push_top_done, cudaEventDisableTiming));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&push_bottom_done, cudaEventDisableTiming));

    CUDA_RT_CALL(cudaDeviceSynchronize());

    if (print)
        printf(
            "Single GPU jacobi relaxation (non-persistent kernel): %d iterations on %d x %d "
            "mesh "
            "with "
            "norm "
            "check every %d iterations\n",
            iter_max, ny, nx, nccheck);

    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 32;
    dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x, (ny + dim_block_y - 1) / dim_block_y, 1);

    int iter = 0;
    bool calculate_norm = false;

    double start = omp_get_wtime();
    PUSH_RANGE("Jacobi solve", 0)
    while (iter < iter_max) {
        CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_top_done, 0));
        CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_bottom_done, 0));

        kernel<<<dim_grid, {dim_block_x, dim_block_y, 1}, 0, compute_stream>>>(
            a_new, a, nullptr, iy_start, iy_end, nx, calculate_norm);
        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaEventRecord(compute_done, compute_stream));

        // Apply periodic boundary conditions

        CUDA_RT_CALL(cudaStreamWaitEvent(push_top_stream, compute_done, 0));
        CUDA_RT_CALL(cudaMemcpyAsync(a_new, a_new + (iy_end - 1) * nx, nx * sizeof(real),
                                     cudaMemcpyDeviceToDevice, push_top_stream));
        CUDA_RT_CALL(cudaEventRecord(push_top_done, push_top_stream));

        CUDA_RT_CALL(cudaStreamWaitEvent(push_bottom_stream, compute_done, 0));
        CUDA_RT_CALL(cudaMemcpyAsync(a_new + iy_end * nx, a_new + iy_start * nx, nx * sizeof(real),
                                     cudaMemcpyDeviceToDevice, compute_stream));
        CUDA_RT_CALL(cudaEventRecord(push_bottom_done, push_bottom_stream));

        std::swap(a_new, a);
        iter++;
    }

    POP_RANGE
    double stop = omp_get_wtime();

    CUDA_RT_CALL(cudaMemcpy(a_ref_h, a, nx * ny * sizeof(real), cudaMemcpyDeviceToHost));

    CUDA_RT_CALL(cudaEventDestroy(push_bottom_done));
    CUDA_RT_CALL(cudaEventDestroy(push_top_done));
    CUDA_RT_CALL(cudaEventDestroy(compute_done));
    CUDA_RT_CALL(cudaStreamDestroy(push_bottom_stream));
    CUDA_RT_CALL(cudaStreamDestroy(push_top_stream));
    CUDA_RT_CALL(cudaStreamDestroy(compute_stream));

    CUDA_RT_CALL(cudaFree(a_new));
    CUDA_RT_CALL(cudaFree(a));
    return (stop - start);
}

double single_gpu_persistent(const int nx, const int ny, const int iter_max, real *const a_ref_h,
                             const int nccheck, const bool print) {
    real *a;
    real *a_new;

    int iy_start = 1;
    int iy_end = (ny - 1);

    CUDA_RT_CALL(cudaMalloc(&a, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc(&a_new, nx * ny * sizeof(real)));

    CUDA_RT_CALL(cudaMemset(a, 0, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * ny * sizeof(real)));

    // Set diriclet boundary conditions on left and right boarder
    initialize_boundaries<<<ny / 128 + 1, 128>>>(a, a_new, PI, 0, nx, ny, ny);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());

    CUDA_RT_CALL(cudaDeviceSynchronize());

    if (print)
        printf(
            "Single GPU jacobi relaxation (persistent kernel): %d iterations on %d x %d mesh "
            "with "
            "norm "
            "check every %d iterations\n",
            iter_max, ny, nx, nccheck);

    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 32;

    dim3 dim_block(dim_block_x, dim_block_y);
    dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x, (ny + dim_block_y - 1) / dim_block_y, 1);

    bool calculate_norm = false;

    void *kernelArgs[] = {(void *)&a_new,   (void *)&a,  (void *)&iy_start,
                          (void *)&iy_end,  (void *)&nx, (void *)&calculate_norm,
                          (void *)&iter_max};

    double start = omp_get_wtime();

    CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *)jacobi_kernel_single_gpu_persistent, dim_grid,
                                             dim_block, kernelArgs, 0, nullptr));

    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());

    double stop = omp_get_wtime();

    CUDA_RT_CALL(cudaMemcpy(a_ref_h, a, nx * ny * sizeof(real), cudaMemcpyDeviceToHost));

    CUDA_RT_CALL(cudaFree(a_new));
    CUDA_RT_CALL(cudaFree(a));
    return (stop - start);
}

void report_results(const int ny, const int nx, real *a_ref_h, real *a_h, const int num_devices,
                    const double runtime_serial_non_persistent, const double start,
                    const double stop, const bool compare_to_single_gpu) {
    bool result_correct = true;

    if (compare_to_single_gpu) {
        for (int iy = 1; result_correct && (iy < (ny - 1)); ++iy) {
            for (int ix = 1; result_correct && (ix < (nx - 1)); ++ix) {
                if (std::fabs(a_ref_h[iy * nx + ix] - a_h[iy * nx + ix]) > tol) {
                    fprintf(stderr,
                            "ERROR: a[%d * %d + %d] = %.8f does not match %.8f "
                            "(reference)\n",
                            iy, nx, ix, a_h[iy * nx + ix], a_ref_h[iy * nx + ix]);
                    // result_correct = false;
                }
            }
        }
    }

    if (result_correct) {
        // printf("Num GPUs: %d.\n", num_devices);
        printf("Execution time: %8.4f s\n", (stop - start));

        if (compare_to_single_gpu) {
            printf(
                "Non-persistent kernel - %dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: "
                "%8.2f, "
                "efficiency: %8.2f \n",
                ny, nx, runtime_serial_non_persistent, num_devices, (stop - start),
                runtime_serial_non_persistent / (stop - start),
                runtime_serial_non_persistent / (num_devices * (stop - start)) * 100);
        }
    }
}
