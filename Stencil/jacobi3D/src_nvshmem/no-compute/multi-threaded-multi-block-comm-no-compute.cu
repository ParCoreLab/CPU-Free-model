/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */
#include "../../include_nvshmem/no-compute/multi-threaded-multi-block-comm-no-compute.cuh"

namespace cg = cooperative_groups;

namespace SSMultiThreadedMultiBlockCommNvshmemNoCompute {

__global__ void __launch_bounds__(1024, 1)
    jacobi_kernel(real *a_new, real *a, const int iz_start, const int iz_end, const int ny,
                  const int nx, const int comm_sm_count_per_layer,
                  const int comm_block_count_per_sm, const int comp_block_count_per_sm,
                  const int tile_count_y, const int tile_count_x, const int iter_max,
                  real *halo_buffer_top, real *halo_buffer_bottom,
                  uint64_t *is_done_computing_flags, const int top, const int bottom) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    int iter = 0;
    int cur_iter_mod = 0;
    int next_iter_mod = 1;

    // const int comp_start_iz = ((blockIdx.x * comp_block_count_per_sm / (tile_count_x *
    // tile_count_y)) * blockDim.z + threadIdx.z + iz_start + 1); const int comp_start_iy =
    // ((((blockIdx.x * comp_block_count_per_sm) / tile_count_x) % tile_count_y) * blockDim.y +
    // threadIdx.y + 1); const int comp_start_ix = ((blockIdx.x * comp_block_count_per_sm) %
    // tile_count_x) * blockDim.x + threadIdx.x + 1;

    const int comm_block_id = (gridDim.x - 1) - blockIdx.x;
    const int comm_start_block_y =
        (((comm_block_id * comm_block_count_per_sm) / tile_count_x) * blockDim.y);
    const int comm_start_block_x =
        ((comm_block_id * comm_block_count_per_sm) % tile_count_x) * blockDim.x;
    //    const int comm_start_iy = comm_start_block_y + (threadIdx.y + 1);
    //    const int comm_start_ix = comm_start_block_x + threadIdx.x + 1;

    while (iter < iter_max) {
        if (comm_block_id < comm_sm_count_per_layer) {
            if (!cta.thread_rank()) {
                nvshmem_signal_wait_until(is_done_computing_flags +
                                              cur_iter_mod * 2 * comm_sm_count_per_layer +
                                              comm_block_id,
                                          NVSHMEM_CMP_EQ, iter);
            }
            /*
            cg::sync(cta);

            int block_count = 0;
            int iy = comm_start_iy;
            int ix = comm_start_ix;

            for (; block_count < comm_block_count_per_sm && iy < (ny - 1); iy += blockDim.y)
            {
                for (; block_count < comm_block_count_per_sm && ix < (nx - 1); ix += blockDim.x)
                {
                    const real first_row_val = (real(1) / real(6)) * (a[iz_start * ny * nx + iy * nx
            + ix + 1] + a[iz_start * ny * nx + iy * nx + ix - 1] + a[iz_start * ny * nx + (iy + 1) *
            nx + ix] + a[iz_start * ny * nx + (iy - 1) * nx + ix] + a[(iz_start + 1) * ny * nx + iy
            * nx + ix] + halo_buffer_top[cur_iter_mod * ny * nx + iy * nx + ix]); a_new[iz_start *
            ny * nx + iy * nx + ix] = first_row_val; block_count++;
                }
                block_count += (block_count < comp_block_count_per_sm) && !(ix < (nx - 1));
                ix = (threadIdx.x + 1);
            }
            */

            nvshmemx_putmem_signal_nbi_block(
                halo_buffer_bottom + next_iter_mod * ny * nx + comm_start_block_y * nx +
                    comm_start_block_x,
                a_new + iz_start * ny * nx + comm_start_block_y * nx + comm_start_block_x,
                min(comm_block_count_per_sm * cta.num_threads(),
                    ny * nx - comm_start_block_y * nx - comm_start_block_x) *
                    sizeof(real),
                is_done_computing_flags + next_iter_mod * 2 * comm_sm_count_per_layer +
                    comm_sm_count_per_layer + comm_block_id,
                iter + 1, NVSHMEM_SIGNAL_SET, top);
        } else if (comm_block_id < 2 * comm_sm_count_per_layer) {
            if (!cta.thread_rank()) {
                nvshmem_signal_wait_until(
                    is_done_computing_flags + cur_iter_mod * 2 * comm_sm_count_per_layer +
                        comm_sm_count_per_layer + comm_block_id - comm_sm_count_per_layer,
                    NVSHMEM_CMP_EQ, iter);
            }
            /*
            cg::sync(cta);

            int block_count = 0;
            int iy = comm_start_iy;
            int ix = comm_start_ix;

            for (; block_count < comm_block_count_per_sm && iy < (ny - 1); iy += blockDim.y)
            {
                for (; block_count < comm_block_count_per_sm && ix < (nx - 1); ix += blockDim.x)
                {
                    const real last_row_val = (real(1) / real(6)) * (a[(iz_end - 1) * ny * nx + iy *
            nx + ix + 1] + a[(iz_end - 1) * ny * nx + iy * nx + ix - 1] + a[(iz_end - 1) * ny * nx +
            (iy + 1) * nx + ix] + a[(iz_end - 1) * ny * nx + (iy - 1) * nx + ix] +
                                                                     halo_buffer_bottom[cur_iter_mod
            * ny * nx + iy * nx + ix] + a[(iz_end - 2) * ny * nx + iy * nx + ix]); a_new[(iz_end -
            1) * ny * nx + iy * nx + ix] = last_row_val; block_count++;
                }
                block_count += (block_count < comp_block_count_per_sm) && !(ix < (nx - 1));
                ix = (threadIdx.x + 1);
            }
            */

            nvshmemx_putmem_signal_nbi_block(
                halo_buffer_top + next_iter_mod * ny * nx + comm_start_block_y * nx +
                    comm_start_block_x,
                a_new + (iz_end - 1) * ny * nx + comm_start_block_y * nx + comm_start_block_x,
                min(comm_block_count_per_sm * cta.num_threads(),
                    ny * nx - comm_start_block_y * nx - comm_start_block_x) *
                    sizeof(real),
                is_done_computing_flags + next_iter_mod * 2 * comm_sm_count_per_layer +
                    comm_block_id - comm_sm_count_per_layer,
                iter + 1, NVSHMEM_SIGNAL_SET, bottom);
        } else {
            /*
                int iz = comp_start_iz;
                int iy = comp_start_iy;
                int ix = comp_start_ix;
                int block_count = 0;

                for (; block_count < comp_block_count_per_sm && iz < (iz_end - 1); iz += blockDim.z)
                {
                    for (; block_count < comp_block_count_per_sm && iy < (ny - 1); iy += blockDim.y)
                    {
                        for (; block_count < comp_block_count_per_sm && ix < (nx - 1); ix +=
               blockDim.x)
                        {
                            a_new[iz * ny * nx + iy * nx + ix] = (real(1) / real(6)) *
                                                                 (a[iz * ny * nx + iy * nx + ix + 1]
               + a[iz * ny * nx + iy * nx + ix - 1] + a[iz * ny * nx + (iy + 1) * nx + ix] + a[iz *
               ny * nx + (iy - 1) * nx + ix] + a[(iz + 1) * ny * nx + iy * nx + ix] + a[(iz - 1) *
               ny * nx + iy * nx + ix]); block_count++;
                        }
                        block_count += (block_count < comp_block_count_per_sm) && !(ix < (nx - 1));
                        ix = (threadIdx.x + 1);
                    }
                    iy = (threadIdx.y + 1);
                }
            */
        }

        real *temp_pointer = a_new;
        a_new = a;
        a = temp_pointer;

        iter++;

        next_iter_mod = cur_iter_mod;
        cur_iter_mod = 1 - cur_iter_mod;

        cg::sync(grid);
    }
}
}  // namespace SSMultiThreadedMultiBlockCommNvshmemNoCompute

int SSMultiThreadedMultiBlockCommNvshmemNoCompute::init(int argc, char *argv[]) {
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
    long long unsigned int mesh_size_per_rank = nx * ny * 3;
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

    constexpr int num_threads_per_block = 1024;
    int device;
    CUDA_RT_CALL(cudaGetDevice(&device));
    cudaDeviceProp deviceProp{};
    CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, device));
    int numSms = deviceProp.multiProcessorCount;

    const int dim_block_x =
        nx >= num_threads_per_block ? num_threads_per_block : (int)pow(2, ceil(log2(nx)));
    const int dim_block_y = ny >= (num_threads_per_block / dim_block_x)
                                ? (num_threads_per_block / dim_block_x)
                                : (int)pow(2, ceil(log2(ny)));
    const int dim_block_z = chunk_size >= (num_threads_per_block / (dim_block_x * dim_block_y))
                                ? (num_threads_per_block / (dim_block_x * dim_block_y))
                                : (int)pow(2, ceil(log2(ny)));

    const int tile_count_x = nx / (dim_block_x) + (nx % (dim_block_x) != 0);
    const int tile_count_y = ny / (dim_block_y) + (ny % (dim_block_y) != 0);
    const int tile_count_z = chunk_size / (dim_block_z) + (chunk_size % (dim_block_z) != 0);

    const int comm_layer_tile_count = tile_count_x * tile_count_y;
    const int comp_total_tile_count = tile_count_x * tile_count_y * tile_count_z;

    const int comm_sm_count_per_layer = 1;
    const int comp_sm_count = comp_total_tile_count < numSms - 2 * comm_sm_count_per_layer
                                  ? comp_total_tile_count
                                  : numSms - 2 * comm_sm_count_per_layer;

    int total_num_flags = 2 * 2 * comm_sm_count_per_layer;

    const int comp_block_count_per_sm =
        comp_total_tile_count / comp_sm_count + (comp_total_tile_count % comp_sm_count != 0);
    const int comm_block_count_per_sm = comm_layer_tile_count / comm_sm_count_per_layer +
                                        (comm_layer_tile_count % comm_sm_count_per_layer != 0);

    const int top_pe = mype > 0 ? mype - 1 : (npes - 1);
    const int bottom_pe = (mype + 1) % npes;

    nvshmem_barrier_all();

    CUDA_RT_CALL(cudaMalloc(&a, nx * ny * (chunk_size + 2) * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc(&a_new, nx * ny * (chunk_size + 2) * sizeof(real)));

    CUDA_RT_CALL(cudaMemset(a, 0, nx * ny * (chunk_size + 2) * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * ny * (chunk_size + 2) * sizeof(real)));

    halo_buffer_top = (real *)nvshmem_malloc(2 * nx * ny * sizeof(real));
    halo_buffer_bottom = (real *)nvshmem_malloc(2 * nx * ny * sizeof(real));

    CUDA_RT_CALL(cudaMemset((void *)halo_buffer_top, 0, 2 * nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMemset((void *)halo_buffer_bottom, 0, 2 * nx * ny * sizeof(real)));

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

    initialize_boundaries<<<(nz / npes) / 128 + 1, 128>>>(a_new, a, PI, iz_start_global - 1, nx, ny,
                                                          chunk_size + 2, nz);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());

    dim3 dim_grid(comp_sm_count + comm_sm_count_per_layer * 2);
    dim3 dim_block(dim_block_x, dim_block_y, dim_block_z);

    void *kernelArgs[] = {(void *)&a_new,
                          (void *)&a,
                          (void *)&iz_start,
                          (void *)&iz_end,
                          (void *)&ny,
                          (void *)&nx,
                          (void *)&comm_sm_count_per_layer,
                          (void *)&comm_block_count_per_sm,
                          (void *)&comp_block_count_per_sm,
                          (void *)&tile_count_y,
                          (void *)&tile_count_x,
                          (void *)&iter_max,
                          (void *)&halo_buffer_top,
                          (void *)&halo_buffer_bottom,
                          (void *)&is_done_computing_flags,
                          (void *)&top_pe,
                          (void *)&bottom_pe};

    nvshmem_barrier_all();
    double start = MPI_Wtime();

    CUDA_RT_CALL((cudaError_t)nvshmemx_collective_launch(
        (void *)SSMultiThreadedMultiBlockCommNvshmemNoCompute::jacobi_kernel, dim_grid, dim_block,
        kernelArgs, 0, nullptr));

    CUDA_RT_CALL(cudaDeviceSynchronize());
    CUDA_RT_CALL(cudaGetLastError());
    // Need to swap pointers on CPU if iteration count is odd
    // Technically, we don't know the iteration number (since we'll be doing
    // l2-norm) Could write iter to CPU when kernel is done
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
