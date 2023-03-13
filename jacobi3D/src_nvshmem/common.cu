#include <assert.h>
#include <cooperative_groups.h>

#include "../include_nvshmem/common.h"

namespace cg = cooperative_groups;

bool get_arg(char **begin, char **end, const std::string &arg) {
    char **itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

__global__ void initialize_boundaries(real *__restrict__ const a_new, real *__restrict__ const a,
                                      const real pi, const int offset, const int nx, const int ny,
                                      const int my_nz, const int nz) {
    for (unsigned int iz = blockIdx.x * blockDim.x + threadIdx.x; iz < my_nz;
         iz += blockDim.x * gridDim.x) {
        const real y0 = sin(2.0 * pi * (offset + iz) / (nz - 1));
        for (unsigned int iy = 0; iy < ny; iy++) {
            for (unsigned int ix = 0; ix < nx; ix++) {
                a[iz * ny * nx + iy * nx + ix] = y0;
                a_new[iz * ny * nx + iy * nx + ix] = y0;

                //                a[iz * ny * nx + iy * nx + ix] = iz * ny * nx + iy * nx + ix;
                //                a_new[iz * ny * nx + iy * nx + ix] = iz * ny * nx + iy * nx + ix;
                //                // + 0.5;
            }
        }
    }
}

__global__ void jacobi_kernel_single_gpu(real *__restrict__ const a_new,
                                         const real *__restrict__ const a,
                                         real *__restrict__ const l2_norm, const int iz_start,
                                         const int iz_end, const int ny, const int nx,
                                         const bool calculate_norm) {
    int iz = blockIdx.z * blockDim.z + threadIdx.z + iz_start;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (iz < iz_end && iy < (ny - 1) && ix < (nx - 1)) {
        const real new_val =
            (real(1) / real(6)) *
            (a[iz * ny * nx + iy * nx + ix + 1] + a[iz * ny * nx + iy * nx + ix - 1] +
             a[iz * ny * nx + (iy + 1) * nx + ix] + a[iz * ny * nx + (iy - 1) * nx + ix] +
             a[(iz + 1) * ny * nx + iy * nx + ix] + a[(iz - 1) * ny * nx + iy * nx + ix]);

        a_new[iz * ny * nx + iy * nx + ix] = new_val;
    }
}

__global__ void jacobi_kernel_single_gpu_mirror(real *__restrict__ const a_new,
                                                const real *__restrict__ const a,
                                                real *__restrict__ const l2_norm,
                                                const int iz_start, const int iz_end, const int ny,
                                                const int nx, const bool calculate_norm) {
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    // WE COMPUTE 1 AND 244 YOU
    if (iz < iz_start || iz > (iz_end - 1)) {
        return;
    }

    real east = a[iz * ny * nx + iy * nx + ix + (ix < (nx - 1))];
    real north = a[iz * ny * nx + (iy + (iy < (ny - 1))) * nx + ix];

    real west = a[iz * ny * nx + iy * nx + ix - (ix > 0)];
    real south = a[iz * ny * nx + (iy - (iy > 0)) * nx + ix];

    real top = a[(iz - 1) * ny * nx + iy * nx + ix];
    real bottom = a[(iz + 1) * ny * nx + iy * nx + ix];

    const real new_val = (1.0f / 6.0f) * (north + south + west + east + top + bottom);

    a_new[iz * ny * nx + iy * nx + ix] = new_val;
}

__global__ void jacobi_kernel_single_gpu_persistent(real *a_new, real *a, const int iz_start,
                                                    const int iz_end, const int ny, const int nx,
                                                    const bool calculate_norm, const int iter_max) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    int iz = blockIdx.z * blockDim.z + threadIdx.z + iz_start;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;

    //    real local_l2_norm = 0.0;

    int iter = 0;

    while (iter < iter_max) {
        if (iz < iz_end && iy < (ny - 1) && ix < (nx - 1)) {
            const real new_val =
                (real(1) / real(6)) *
                (a[iz * ny * nx + iy * nx + ix + 1] + a[iz * ny * nx + iy * nx + ix - 1] +
                 a[iz * ny * nx + (iy + 1) * nx + ix] + a[iz * ny * nx + (iy - 1) * nx + ix] +
                 a[(iz + 1) * ny * nx + iy * nx + ix] + a[(iz - 1) * ny * nx + iy * nx + ix]);
            a_new[iz * ny * nx + iy * nx + ix] = new_val;

            if (iz_start == iz) {
                a_new[iz_end * ny * nx + iy * nx + ix] = new_val;
            }

            if ((iz_end - 1) == iz) {
                a_new[(iz_start - 1) * ny * nx + iy * nx + ix] = new_val;
            }

            //        if (calculate_norm) {
            //            real residue = new_val - a[iy * nx + ix];
            //            local_l2_norm += residue * residue;
            //        }
        }

        iter++;

        real *temp_pointer = a_new;
        a_new = a;
        a = temp_pointer;

        cg::sync(grid);
    }

    //    if (calculate_norm) {
    //        atomicAdd(l2_norm, local_l2_norm);
    //    }
}

double single_gpu(const int nz, const int ny, const int nx, const int iter_max, real *const a_ref_h,
                  const int nccheck, const bool print, decltype(jacobi_kernel_single_gpu) kernel) {
    real *a;
    real *a_new;

    cudaStream_t compute_stream;
    cudaStream_t push_top_stream;
    cudaStream_t push_bottom_stream;
    cudaEvent_t compute_done;
    cudaEvent_t push_top_done;
    cudaEvent_t push_bottom_done;

    //    real* l2_norm_d;
    //    real* l2_norm_h;

    int iz_start = 1;
    int iz_end = (nz - 1);

    CUDA_RT_CALL(cudaMalloc(&a, nx * ny * nz * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc(&a_new, nx * ny * nz * sizeof(real)));

    CUDA_RT_CALL(cudaMemset(a, 0, nx * ny * nz * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * ny * nz * sizeof(real)));

    // Set diriclet boundary conditions on left and right boarder
    initialize_boundaries<<<nz / 128 + 1, 128>>>(a_new, a, PI, 0, nx, ny, nz, nz);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());

    CUDA_RT_CALL(cudaStreamCreate(&compute_stream));
    CUDA_RT_CALL(cudaStreamCreate(&push_top_stream));
    CUDA_RT_CALL(cudaStreamCreate(&push_bottom_stream));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&compute_done, cudaEventDisableTiming));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&push_top_done, cudaEventDisableTiming));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&push_bottom_done, cudaEventDisableTiming));

    //    CUDA_RT_CALL(cudaMalloc(&l2_norm_d, sizeof(real)));
    //    CUDA_RT_CALL(cudaMallocHost(&l2_norm_h, sizeof(real)));

    CUDA_RT_CALL(cudaDeviceSynchronize());

    if (print)
        printf(
            "Single GPU jacobi relaxation (non-persistent kernel): %d "
            "iterations on %d x %d x %d "
            "mesh "
            "with "
            "norm "
            "check every %d iterations\n",
            iter_max, nx, ny, nz, nccheck);
    fflush(stdout);
    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 8;
    constexpr int dim_block_z = 4;

    dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x, (ny + dim_block_y - 1) / dim_block_y,
                  (nz + dim_block_z - 1) / dim_block_z);

    int iter = 0;
    bool calculate_norm = false;
    //    real l2_norm = 1.0;

    double start = omp_get_wtime();
    PUSH_RANGE("Jacobi solve", 0)

    while (iter < iter_max) {
        //        CUDA_RT_CALL(cudaMemsetAsync(l2_norm_d, 0, sizeof(real),
        //        compute_stream));

        CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_top_done, 0));
        CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_bottom_done, 0));

        //        calculate_norm = (iter % nccheck) == 0 || (print && ((iter %
        //        100)
        //        == 0));
        kernel<<<dim_grid, {dim_block_x, dim_block_y, dim_block_z}, 0, compute_stream>>>(
            a_new, a, nullptr, iz_start, iz_end, ny, nx, calculate_norm);
        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaEventRecord(compute_done, compute_stream));

        //        if (calculate_norm) {
        //            CUDA_RT_CALL(cudaMemcpyAsync(l2_norm_h, l2_norm_d,
        //            sizeof(real), cudaMemcpyDeviceToHost,
        //                                         compute_stream));
        //        }

        // Apply periodic boundary conditions

        CUDA_RT_CALL(cudaStreamWaitEvent(push_top_stream, compute_done, 0));
        CUDA_RT_CALL(cudaMemcpyAsync(a_new, a_new + (iz_end - 1) * ny * nx, nx * ny * sizeof(real),
                                     cudaMemcpyDeviceToDevice, push_top_stream));
        CUDA_RT_CALL(cudaEventRecord(push_top_done, push_top_stream));

        CUDA_RT_CALL(cudaStreamWaitEvent(push_bottom_stream, compute_done, 0));
        CUDA_RT_CALL(cudaMemcpyAsync(a_new + iz_end * ny * nx, a_new + iz_start * ny * nx,
                                     nx * ny * sizeof(real), cudaMemcpyDeviceToDevice,
                                     compute_stream));
        CUDA_RT_CALL(cudaEventRecord(push_bottom_done, push_bottom_stream));

        //        if (calculate_norm) {
        //            CUDA_RT_CALL(cudaStreamSynchronize(compute_stream));
        //            l2_norm = *l2_norm_h;
        //            l2_norm = std::sqrt(l2_norm);
        //            if (print && (iter % 100) == 0) printf("%5d, %0.6f\n",
        //            iter, l2_norm);
        //        }

        std::swap(a_new, a);
        iter++;
    }

    CUDA_RT_CALL(cudaDeviceSynchronize());

    POP_RANGE
    double stop = omp_get_wtime();

    CUDA_RT_CALL(cudaMemcpy(a_ref_h, a, nx * ny * nz * sizeof(real), cudaMemcpyDeviceToHost));

    CUDA_RT_CALL(cudaEventDestroy(push_bottom_done));
    CUDA_RT_CALL(cudaEventDestroy(push_top_done));
    CUDA_RT_CALL(cudaEventDestroy(compute_done));
    CUDA_RT_CALL(cudaStreamDestroy(push_bottom_stream));
    CUDA_RT_CALL(cudaStreamDestroy(push_top_stream));
    CUDA_RT_CALL(cudaStreamDestroy(compute_stream));

    //    CUDA_RT_CALL(cudaFreeHost(l2_norm_h));
    //    CUDA_RT_CALL(cudaFree(l2_norm_d));

    CUDA_RT_CALL(cudaFree(a_new));
    CUDA_RT_CALL(cudaFree(a));

    return (stop - start);
}

double single_gpu_persistent(const int nz, const int ny, const int nx, const int iter_max,
                             real *const a_ref_h, const int nccheck, const bool print) {
    real *a;
    real *a_new;

    // Skipping l2-norm calculation for now
    //    real* l2_norm_d;
    //    real* l2_norm_h;

    int iz_start = 1;
    int iz_end = (nz - 1);

    CUDA_RT_CALL(cudaMalloc(&a, nx * ny * nz * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc(&a_new, nx * ny * nz * sizeof(real)));

    CUDA_RT_CALL(cudaMemset(a, 0, nx * ny * nz * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * ny * nz * sizeof(real)));

    // Set diriclet boundary conditions on left and right boarder
    initialize_boundaries<<<ny / 128 + 1, 128>>>(a_new, a, PI, 0, nx, ny, nz, nz);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());

    //    CUDA_RT_CALL(cudaMalloc(&l2_norm_d, sizeof(real)));
    //    CUDA_RT_CALL(cudaMallocHost(&l2_norm_h, sizeof(real)));

    CUDA_RT_CALL(cudaDeviceSynchronize());

    if (print)
        printf(
            "Single GPU jacobi relaxation (persistent kernel): %d iterations on "
            "%d x %d x %d mesh "
            "with "
            "norm "
            "check every %d iterations\n",
            iter_max, nx, ny, nz, nccheck);

    constexpr int dim_block_x = 8;
    constexpr int dim_block_y = 8;
    constexpr int dim_block_z = 16;

    dim3 dim_block(dim_block_x, dim_block_y, dim_block_z);
    dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x, (ny + dim_block_y - 1) / dim_block_y,
                  (nz + dim_block_z - 1) / dim_block_z);

    bool calculate_norm = false;
    //    real l2_norm = 1.0;

    void *kernelArgs[] = {(void *)&a_new, (void *)&a,  (void *)&iz_start,       (void *)&iz_end,
                          (void *)&ny,    (void *)&nx, (void *)&calculate_norm, (void *)&iter_max};

    double start = omp_get_wtime();

    CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *)jacobi_kernel_single_gpu_persistent, dim_grid,
                                             dim_block, kernelArgs, 0, nullptr));

    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());

    double stop = omp_get_wtime();

    CUDA_RT_CALL(cudaMemcpy(a_ref_h, a, nx * ny * nz * sizeof(real), cudaMemcpyDeviceToHost));

    //    CUDA_RT_CALL(cudaFreeHost(l2_norm_h));
    //    CUDA_RT_CALL(cudaFree(l2_norm_d));

    CUDA_RT_CALL(cudaFree(a_new));
    CUDA_RT_CALL(cudaFree(a));
    return (stop - start);
}

void report_results(const int nz, const int ny, const int nx, real *a_ref_h, real *a_h,
                    const int num_devices, const double runtime_serial_non_persistent,
                    const double start, const double stop, const bool compare_to_single_gpu) {
    bool result_correct = true;

    if (compare_to_single_gpu) {
        for (int iz = 1; result_correct && (iz < (nz - 1)); ++iz) {
            for (int iy = 1; result_correct && (iy < (ny - 1)); ++iy) {
                for (int ix = 1; result_correct && (ix < (nx - 1)); ++ix) {
                    if (std::fabs(a_h[iz * ny * nx + iy * nx + ix] -
                                  a_ref_h[iz * ny * nx + iy * nx + ix]) > tol) {
                        fprintf(stderr,
                                "ERROR: a[%d * %d + %d * %d + %d] = %f does "
                                "not match %f "
                                "(reference)\n",
                                iz, ny * nx, iy, nx, ix, a_h[iz * ny * nx + iy * nx + ix],
                                a_ref_h[iz * ny * nx + iy * nx + ix]);
                        // result_correct = false;
                    }
                }
            }
        }
    }

    if (result_correct) {
        // printf("Num GPUs: %d.\n", num_devices);
        printf("Execution time: %8.4f s\n", (stop - start));

        if (compare_to_single_gpu) {
            printf(
                "Non-persistent kernel - %dx%dx%d: 1 GPU: %8.4f s, %d GPUs: "
                "%8.4f "
                "s, speedup: "
                "%8.2f, "
                "efficiency: %8.2f \n",
                nx, ny, nz, runtime_serial_non_persistent, num_devices, (stop - start),
                runtime_serial_non_persistent / (stop - start),
                runtime_serial_non_persistent / (num_devices * (stop - start)) * 100);
        }
    }
}

// convert NVSHMEM_SYMMETRIC_SIZE string to long long unsigned int
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