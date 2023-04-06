/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "../../include_nvshmem/no-compute/multi-gpu-peer-tiling-no-compute.cuh"

namespace cg = cooperative_groups;

namespace MultiGPUPeerTilingNvshmemNoCompute {
__global__ void __launch_bounds__(1024, 1)
    jacobi_kernel(real *a_new, real *a, const int iz_start, const int iz_end, const int ny,
                  const int nx, const int iter_max, volatile int *iteration_done) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    int iter = 0;

    // const int comp_size_iz = gridDim.z * blockDim.z * ny * nx;
    // const int comp_size_iy = gridDim.y * blockDim.y * nx;
    // const int comp_size_ix = gridDim.x * blockDim.x;

    // const int comp_start_iz = (blockIdx.z * blockDim.z + threadIdx.z + iz_start + 1) * ny * nx;
    // const int comp_start_iy = (blockIdx.y * blockDim.y + threadIdx.y + 1) * nx;
    // const int comp_start_ix = blockIdx.x * blockDim.x + threadIdx.x + 1;

    // const int end_iz = (iz_end - 1) * ny * nx;
    // const int end_iy = (ny - 1) * nx;
    // const int end_ix = (nx - 1);

    while (iter < iter_max) {
        /*
        for (int iz = comp_start_iz; iz < end_iz; iz += comp_size_iz)
        {
            for (int iy = comp_start_iy; iy < end_iy; iy += comp_size_iy)
            {
                for (int ix = comp_start_ix; ix < end_ix; ix += comp_size_ix)
                {
                    a_new[iz + iy + ix] = (real(1) / real(6)) * (a[iz + iy + ix + 1] + a[iz + iy +
        ix - 1] + a[iz + iy + nx + ix] + a[iz + iy - nx + ix] + a[iz + ny * nx + iy + ix] + a[iz -
        ny * nx + iy + ix]);
                }
            }
        }
        */

        real *temp_pointer = a_new;
        a_new = a;
        a = temp_pointer;

        iter++;
        cg::sync(grid);
        if (!grid.thread_rank()) {
            while (iteration_done[0] != iter) {
            }
            iteration_done[1] = iter;
        }
        cg::sync(grid);
    }
}

__global__ void __launch_bounds__(1024, 1)
    boundary_sync_kernel(real *a_new, real *a, const int iz_start, const int iz_end, const int ny,
                         const int nx, const int iter_max, real *halo_buffer_top,
                         real *halo_buffer_bottom, uint64_t *is_done_computing_flags, const int top,
                         const int bottom, volatile int *iteration_done) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    int iter = 0;
    int cur_iter_mod = 0;
    int next_iter_mod = 1;

    const int end_iz = (iz_end - 1) * ny * nx;
    // const int end_iy = (ny - 1) * nx;
    // const int end_ix = (nx - 1);

    // const int comm_size_iy = blockDim.y * nx;
    // const int comm_size_ix = blockDim.x;

    // const int comm_start_iy = (threadIdx.y + 1) * nx;
    // const int comm_start_ix = threadIdx.x + 1;
    const int comm_start_iz = iz_start * ny * nx;

    while (iter < iter_max) {
        if (!grid.thread_rank()) {
            while (iteration_done[1] != iter) {
            }
        }
        cg::sync(grid);
        if (blockIdx.x == gridDim.x - 1) {
            if (!cta.thread_rank()) {
                nvshmem_signal_wait_until(is_done_computing_flags + cur_iter_mod * 2,
                                          NVSHMEM_CMP_EQ, iter);
            }
            cg::sync(cta);
            /*
            for (int iy = comm_start_iy; iy < end_iy; iy += comm_size_iy)
            {
                for (int ix = comm_start_ix; ix < end_ix; ix += comm_size_ix)
                {
                    const real first_row_val = (real(1) / real(6)) * (a[comm_start_iz + iy + ix + 1]
            + a[comm_start_iz + iy + ix - 1] + a[comm_start_iz + iy + nx + ix] + a[comm_start_iz +
            iy - nx + ix] + a[comm_start_iz + ny * nx + iy + ix] + halo_buffer_top[cur_iter_mod * ny
            * nx + iy + ix]); a_new[comm_start_iz + iy + ix] = first_row_val;
                }
            }
            */

            nvshmemx_putmem_signal_nbi_block(halo_buffer_bottom + next_iter_mod * ny * nx,
                                             a_new + comm_start_iz, ny * nx * sizeof(real),
                                             is_done_computing_flags + next_iter_mod * 2 + 1,
                                             iter + 1, NVSHMEM_SIGNAL_SET, top);

        } else if (blockIdx.x == gridDim.x - 2) {
            if (!cta.thread_rank()) {
                nvshmem_signal_wait_until(is_done_computing_flags + cur_iter_mod * 2 + 1,
                                          NVSHMEM_CMP_EQ, iter);
            }
            cg::sync(cta);
            /*
            for (int iy = comm_start_iy; iy < end_iy; iy += comm_size_iy)
            {
                for (int ix = comm_start_ix; ix < end_ix; ix += comm_size_ix)
                {

                    const real last_row_val = (real(1) / real(6)) * (a[end_iz + iy + ix + 1] +
                                                                     a[end_iz + iy + ix - 1] +
                                                                     a[end_iz + iy + nx + ix] +
                                                                     a[end_iz + iy - nx + ix] +
                                                                     halo_buffer_bottom[cur_iter_mod
            * ny * nx + iy + ix] + a[end_iz - ny * nx + iy + ix]); a_new[end_iz + iy + ix] =
            last_row_val;
                }
            }
            */

            nvshmemx_putmem_signal_nbi_block(
                halo_buffer_top + next_iter_mod * ny * nx, a_new + end_iz, ny * nx * sizeof(real),
                is_done_computing_flags + next_iter_mod * 2, iter + 1, NVSHMEM_SIGNAL_SET, bottom);
        }

        real *temp_pointer_first = a_new;
        a_new = a;
        a = temp_pointer_first;

        iter++;

        next_iter_mod = cur_iter_mod;
        cur_iter_mod = 1 - cur_iter_mod;

        cg::sync(grid);

        if (!grid.thread_rank()) {
            iteration_done[0] = iter;
        }
    }
}
}  // namespace MultiGPUPeerTilingNvshmemNoCompute

int MultiGPUPeerTilingNvshmemNoCompute::init(int argc, char *argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 512);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 512);
    const int nz = get_argval<int>(argv, argv + argc, "-nz", 512);
    const bool compare_to_single_gpu = get_arg(argv, argv + argc, "-compare");

    real *a;
    real *a_new;

    real *halo_buffer_top;
    real *halo_buffer_bottom;

    uint64_t *is_done_computing_flags;
    int *iteration_done_flags;

    real *a_ref_h;
    real *a_h;

    double runtime_serial_non_persistent = 0.0;

    int rank = 0, size = 1;
    MPI_CALL(MPI_Init(&argc, &argv));
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));

    int num_devices = 0;
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
    long long unsigned int mesh_size_per_rank = nx * ny * 2 + 2;
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

    if (compare_to_single_gpu) {
        CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * nz * sizeof(real)));
        CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * nz * sizeof(real)));

        runtime_serial_non_persistent = single_gpu(nz, ny, nx, iter_max, a_ref_h, 0, true);
    }

    nvshmem_barrier_all();

    int chunk_size;
    int chunk_size_low = (nz - 2) / npes;
    int chunk_size_high = chunk_size_low + 1;

    int num_ranks_low = npes * chunk_size_low + npes - (nz - 2);
    if (mype < num_ranks_low)
        chunk_size = chunk_size_low;
    else
        chunk_size = chunk_size_high;

    int device;
    CUDA_RT_CALL(cudaGetDevice(&device));
    cudaDeviceProp deviceProp{};
    CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, device));
    int numSms = deviceProp.multiProcessorCount;

    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 8;
    constexpr int dim_block_z = 4;

    constexpr int grid_dim_x = 2;
    constexpr int grid_dim_y = 4;
    const int grid_dim_z = (numSms - 2) / (grid_dim_x * grid_dim_y);

    int total_num_flags = 4;

    const int top = mype > 0 ? mype - 1 : (npes - 1);
    const int bottom = (mype + 1) % npes;

    nvshmem_barrier_all();

    CUDA_RT_CALL(cudaMalloc(&a, nx * ny * (chunk_size + 2) * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc(&a_new, nx * ny * (chunk_size + 2) * sizeof(real)));

    CUDA_RT_CALL(cudaMemset(a, 0, nx * ny * (chunk_size + 2) * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * ny * (chunk_size + 2) * sizeof(real)));

    halo_buffer_top = (real *)nvshmem_malloc(2 * nx * ny * sizeof(real));
    halo_buffer_bottom = (real *)nvshmem_malloc(2 * nx * ny * sizeof(real));

    CUDA_RT_CALL(cudaMemset((void *)halo_buffer_top, 0, 2 * nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMemset((void *)halo_buffer_bottom, 0, 2 * nx * ny * sizeof(real)));

    CUDA_RT_CALL(cudaMalloc(&iteration_done_flags, 2 * sizeof(int)));
    CUDA_RT_CALL(cudaMemset(iteration_done_flags, 0, 2 * sizeof(int)));

    is_done_computing_flags = (uint64_t *)nvshmem_malloc(total_num_flags * sizeof(uint64_t));
    CUDA_RT_CALL(cudaMemset(is_done_computing_flags, 0, total_num_flags * sizeof(uint64_t)));

    // Calculate local domain boundaries
    int iz_start_global;  // My start index in the global array
    if (mype < num_ranks_low) {
        iz_start_global = mype * chunk_size_low + 1;
    } else {
        iz_start_global =
            num_ranks_low * chunk_size_low + (mype - num_ranks_low) * chunk_size_high + 1;
    }
    int iz_end_global = iz_start_global + chunk_size - 1;  // My last index in the global array

    int iz_start = 1;
    int iz_end = (iz_end_global - iz_start_global + 1) + iz_start;

    // Set diriclet boundary conditions on left and right border
    initialize_boundaries<<<(nz / npes) / 128 + 1, 128>>>(a_new, a, PI, iz_start_global - 1, nx, ny,
                                                          chunk_size + 2, nz);
    CUDA_RT_CALL(cudaGetLastError());

    CUDA_RT_CALL(cudaDeviceSynchronize());

    dim3 comp_dim_grid(grid_dim_x, grid_dim_y, grid_dim_z);
    dim3 comp_dim_block(dim_block_x, dim_block_y, dim_block_z);

    dim3 comm_dim_grid(2);
    dim3 comm_dim_block(dim_block_x, dim_block_y * dim_block_z);

    void *kernelArgsInner[] = {
        (void *)&a_new, (void *)&a,  (void *)&iz_start, (void *)&iz_end,
        (void *)&ny,    (void *)&nx, (void *)&iter_max, (void *)&iteration_done_flags};

    void *kernelArgsBoundary[] = {(void *)&a_new,
                                  (void *)&a,
                                  (void *)&iz_start,
                                  (void *)&iz_end,
                                  (void *)&ny,
                                  (void *)&nx,
                                  (void *)&iter_max,
                                  (void *)&halo_buffer_top,
                                  (void *)&halo_buffer_bottom,
                                  (void *)&is_done_computing_flags,
                                  (void *)&top,
                                  (void *)&bottom,
                                  (void *)&iteration_done_flags};

    nvshmem_barrier_all();
    double start = MPI_Wtime();

    cudaStream_t inner_domain_stream;
    cudaStream_t boundary_sync_stream;

    CUDA_RT_CALL(cudaStreamCreate(&inner_domain_stream));
    CUDA_RT_CALL(cudaStreamCreate(&boundary_sync_stream));

    // THE KERNELS ARE SERIALIZED!
    // perhaps only on V100
    CUDA_RT_CALL(cudaLaunchCooperativeKernel(
        (void *)MultiGPUPeerTilingNvshmemNoCompute::jacobi_kernel, comp_dim_grid, comp_dim_block,
        kernelArgsInner, 0, inner_domain_stream));

    CUDA_RT_CALL((cudaError_t)nvshmemx_collective_launch(
        (void *)MultiGPUPeerTilingNvshmemNoCompute::boundary_sync_kernel, comm_dim_grid,
        comm_dim_block, kernelArgsBoundary, 0, boundary_sync_stream));

    CUDA_RT_CALL(cudaDeviceSynchronize());
    CUDA_RT_CALL(cudaGetLastError());
    // Need to swap pointers on CPU if iteration count is odd
    // Technically, we don't know the iteration number (since we'll be doing l2-norm)
    // Could write iter to CPU when kernel is done
    if (iter_max % 2 == 1) {
        std::swap(a_new, a);
    }

    nvshmem_barrier_all();
    double stop = MPI_Wtime();
    nvshmem_barrier_all();
    bool result_correct = 1;
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
                        // result_correct = 0;
                    }
                }
            }
        }
    }
    int global_result_correct = 1;
    MPI_CALL(MPI_Allreduce(&result_correct, &global_result_correct, 1, MPI_INT, MPI_MIN,
                           MPI_COMM_WORLD));

    if (!mype && global_result_correct) {
        // printf("Num GPUs: %d.\n", npes);
        printf("Execution time: %8.4f s\n", (stop - start));

        if (compare_to_single_gpu) {
            printf(
                "Non-persistent kernel - %dx%dx%d: 1 GPU: %8.4f s, %d GPUs: "
                "%8.4f "
                "s, speedup: "
                "%8.2f, "
                "efficiency: %8.2f \n",
                nx, ny, nz, runtime_serial_non_persistent, npes, (stop - start),
                runtime_serial_non_persistent / (stop - start),
                runtime_serial_non_persistent / (npes * (stop - start)) * 100);
        }
    }

    CUDA_RT_CALL(cudaFree(a_new));
    CUDA_RT_CALL(cudaFree(a));
    CUDA_RT_CALL(cudaFree(iteration_done_flags));
    nvshmem_free((void *)halo_buffer_top);
    nvshmem_free((void *)halo_buffer_bottom);
    nvshmem_free(is_done_computing_flags);

    if (compare_to_single_gpu) {
        CUDA_RT_CALL(cudaFreeHost(a_h));
        CUDA_RT_CALL(cudaFreeHost(a_ref_h));
    }

    nvshmem_finalize();
    MPI_CALL(MPI_Finalize());
    return (result_correct == 1) ? 0 : 1;
}
