/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
// Adapted from
// https://github.com/NVIDIA/multi-gpu-programming-models/blob/master/nvshmem/jacobi.cu

#include "../../include_nvshmem/baseline/multi-threaded-nvshmem.cuh"

namespace BaselineMultiThreadedNvshmem {
template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z>
__global__ void jacobi_kernel(real *__restrict__ const a_new, const real *__restrict__ const a,
                              const int iz_start, const int iz_end, const int ny, const int nx,
                              const int top_pe, const int top_iz, const int bottom_pe,
                              const int bottom_iz) {
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
        if (iz_start == iz) {
            nvshmem_float_p(a_new + top_iz * ny * nx + iy * nx + ix, new_val, top_pe);
        }
        if ((iz_end - 1) == iz) {
            nvshmem_float_p(a_new + bottom_iz * ny * nx + iy * nx + ix, new_val, bottom_pe);
        }
    }
}

}  // namespace BaselineMultiThreadedNvshmem

int BaselineMultiThreadedNvshmem::init(int argc, char *argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 512);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 512);
    const int nz = get_argval<int>(argv, argv + argc, "-nz", 512);
    const bool compare_to_single_gpu = get_arg(argv, argv + argc, "-compare");

    real *a;
    real *a_new;

    real *a_ref_h;
    real *a_h;

    int rank = 0, size = 1;
    MPI_CALL(MPI_Init(&argc, &argv));
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));

    int num_devices;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

    int local_rank = -1;
    int local_size = 1;
    {
        MPI_Comm local_comm;
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                     &local_comm));

        MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));
        MPI_CALL(MPI_Comm_size(local_comm, &local_size));

        MPI_CALL(MPI_Comm_free(&local_comm));
    }
    if (1 < num_devices && num_devices < local_size) {
        fprintf(
            stderr,
            "ERROR Number of visible devices (%d) is less than number of ranks on the node (%d)!\n",
            num_devices, local_size);
        MPI_CALL(MPI_Finalize());
        return 1;
    }
    if (1 == num_devices) {
        // Only 1 device visible, assuming GPU affinity is handled via CUDA_VISIBLE_DEVICES
        CUDA_RT_CALL(cudaSetDevice(0));
    } else {
        CUDA_RT_CALL(cudaSetDevice(local_rank));
    }
    CUDA_RT_CALL(cudaFree(0));

    MPI_Comm mpi_comm;
    nvshmemx_init_attr_t attr;

    mpi_comm = MPI_COMM_WORLD;
    attr.mpi_comm = &mpi_comm;
    // Set symmetric heap size for nvshmem based on problem size
    // Its default value in nvshmem is 1 GB which is not sufficient
    // for large mesh sizes
    long long unsigned int mesh_size_per_rank = nx * ny * (((nz - 2) + size - 1) / size + 2);
    long long unsigned int required_symmetric_heap_size =
        2 * mesh_size_per_rank * sizeof(real) *
        1.1;  // Factor 2 is because 2 arrays are allocated - a and a_new
              // 1.1 factor is just for alignment or other usage

    char *value = getenv("NVSHMEM_SYMMETRIC_SIZE");
    if (value) { /* env variable is set */
        long long unsigned int size_env = parse_nvshmem_symmetric_size(value);
        if (size_env < required_symmetric_heap_size) {
            fprintf(stderr,
                    "ERROR: Minimum NVSHMEM_SYMMETRIC_SIZE = %lluB, Current NVSHMEM_SYMMETRIC_SIZE "
                    "= %s\n",
                    required_symmetric_heap_size, value);
            MPI_CALL(MPI_Finalize());
            return -1;
        }
    } else {
        char symmetric_heap_size_str[100];
        sprintf(symmetric_heap_size_str, "%llu", required_symmetric_heap_size);
        setenv("NVSHMEM_SYMMETRIC_SIZE", symmetric_heap_size_str, 1);
    }
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    int npes = nvshmem_n_pes();
    int mype = nvshmem_my_pe();

    nvshmem_barrier_all();

    cudaStream_t compute_stream;
    cudaEvent_t compute_done[2];

    double runtime_serial_non_persistent = 0.0;
    if (compare_to_single_gpu) {
        CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * nz * sizeof(real)));
        CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * nz * sizeof(real)));

        runtime_serial_non_persistent = single_gpu(nz, ny, nx, iter_max, a_ref_h, 0, true);
    }

    nvshmem_barrier_all();
    // ny - 2 rows are distributed amongst `size` ranks in such a way
    // that each rank gets either (ny - 2) / size or (ny - 2) / size + 1 rows.
    // This optimizes load balancing when (ny - 2) % size != 0
    int chunk_size;
    int chunk_size_low = (nz - 2) / npes;
    int chunk_size_high = chunk_size_low + 1;
    // To calculate the number of ranks that need to compute an extra row,
    // the following formula is derived from this equation:
    // num_ranks_low * chunk_size_low + (size - num_ranks_low) * (chunk_size_low + 1) = ny - 2
    int num_ranks_low = npes * chunk_size_low + npes -
                        (nz - 2);  // Number of ranks with chunk_size = chunk_size_low
    if (mype < num_ranks_low)
        chunk_size = chunk_size_low;
    else
        chunk_size = chunk_size_high;

    a = (real *)nvshmem_malloc(
        nx * ny * (chunk_size_high + 2) *
        sizeof(real));  // Using chunk_size_high so that it is same across all PEs
    a_new = (real *)nvshmem_malloc(nx * ny * (chunk_size_high + 2) * sizeof(real));

    cudaMemset(a, 0, nx * (chunk_size + 2) * sizeof(real));
    cudaMemset(a_new, 0, nx * (chunk_size + 2) * sizeof(real));

    // Calculate local domain boundaries
    int iz_start_global;  // My start index in the global array
    if (mype < num_ranks_low) {
        iz_start_global = mype * chunk_size_low + 1;
    } else {
        iz_start_global =
            num_ranks_low * chunk_size_low + (mype - num_ranks_low) * chunk_size_high + 1;
    }
    int iz_end_global = iz_start_global + chunk_size - 1;  // My last index in the global array
    // do not process boundaries
    iz_end_global = std::min(iz_end_global, nz - 4);

    int iz_start = 1;
    int iz_end = (iz_end_global - iz_start_global + 1) + iz_start;

    // calculate boundary indices for top and bottom boundaries
    int top_pe = mype > 0 ? mype - 1 : (npes - 1);
    int bottom_pe = (mype + 1) % npes;

    int iz_end_top = (top_pe < num_ranks_low) ? chunk_size_low + 1 : chunk_size_high + 1;
    int iz_start_bottom = 0;

    // Set diriclet boundary conditions on left and right boundary
    initialize_boundaries<<<(nz / npes) / 128 + 1, 128>>>(a_new, a, PI, iz_start_global - 1, nx, ny,
                                                          chunk_size + 2, nz);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());

    CUDA_RT_CALL(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&compute_done[0], cudaEventDisableTiming));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&compute_done[1], cudaEventDisableTiming));

    nvshmemx_barrier_all_on_stream(compute_stream);
    CUDA_RT_CALL(cudaDeviceSynchronize());

    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 8;
    constexpr int dim_block_z = 4;

    dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x, (ny + dim_block_y - 1) / dim_block_y,
                  (chunk_size + dim_block_z - 1) / dim_block_z);

    int iter = 0;

    nvshmem_barrier_all();

    double start = MPI_Wtime();
    PUSH_RANGE("Jacobi solve", 0)

    cudaStreamSynchronize(compute_stream);

    while (iter < iter_max) {
        // on new iteration: old current vars are now previous vars, old
        // previous vars are no longer needed
        // int prev = iter % 2;
        // int curr = (iter + 1) % 2;

        jacobi_kernel<dim_block_x, dim_block_y, dim_block_z>
            <<<dim_grid, {dim_block_x, dim_block_y, dim_block_z}, 0, compute_stream>>>(
                a_new, a, iz_start, iz_end, ny, nx, top_pe, iz_end_top, bottom_pe, iz_start_bottom);
        CUDA_RT_CALL(cudaGetLastError());

        nvshmemx_barrier_all_on_stream(compute_stream);

        std::swap(a_new, a);
        iter++;
    }

    CUDA_RT_CALL(cudaDeviceSynchronize());

    nvshmem_barrier_all();
    double stop = MPI_Wtime();
    nvshmem_barrier_all();

    bool result_correct = true;
    if (compare_to_single_gpu) {
        CUDA_RT_CALL(cudaMemcpy(a_h + iz_start_global * ny * nx, a + ny * nx,
                                std::min(nz - iz_start_global, chunk_size) * nx * ny * sizeof(real),
                                cudaMemcpyDeviceToHost));

        for (int iz = iz_start_global; result_correct && (iz <= iz_end_global); ++iz) {
            for (int iy = 1; result_correct && (iy < (ny - 1)); ++iy) {
                for (int ix = 1; result_correct && (ix < (nx - 1)); ++ix) {
                    if (std::fabs(a_h[iz * ny * nx + iy * nx + ix] -
                                  a_ref_h[iz * ny * nx + iy * nx + ix]) > tol) {
                        fprintf(stderr,
                                "ERROR on rank %d: a[%d * %d + %d * %d + %d] = %f does "
                                "not match %f "
                                "(reference)\n",
                                rank, iz, ny * nx, iy, nx, ix, a_h[iz * ny * nx + iy * nx + ix],
                                a_ref_h[iz * ny * nx + iy * nx + ix]);
                        result_correct = 0;
                    }
                }
            }
        }
    }
    int global_result_correct = 1;
    MPI_CALL(MPI_Allreduce(&result_correct, &global_result_correct, 1, MPI_INT, MPI_MIN,
                           MPI_COMM_WORLD));

    if (!mype && global_result_correct) {
        // printf("Num GPUs: %d.\n", num_devices);
        printf("Execution time: %8.4f s\n", (stop - start));

        if (compare_to_single_gpu) {
            printf(
                "Non-persistent kernel - %dx%dx%d: 1 GPU: %8.4f s, %d GPUs: "
                "%8.4f "
                "s, speedup: "
                "%8.2f, "
                "efficiency: %8.2f \n",
                nz, ny, nx, runtime_serial_non_persistent, npes, (stop - start),
                runtime_serial_non_persistent / (stop - start),
                runtime_serial_non_persistent / (npes * (stop - start)) * 100);
        }
    }

    nvshmem_free(a);
    nvshmem_free(a_new);

    CUDA_RT_CALL(cudaEventDestroy(compute_done[1]));
    CUDA_RT_CALL(cudaEventDestroy(compute_done[0]));
    CUDA_RT_CALL(cudaStreamDestroy(compute_stream));

    if (compare_to_single_gpu) {
        CUDA_RT_CALL(cudaFreeHost(a_h));
        CUDA_RT_CALL(cudaFreeHost(a_ref_h));
    }

    nvshmem_finalize();
    MPI_CALL(MPI_Finalize());

    return (result_correct == 1) ? 0 : 1;
}
