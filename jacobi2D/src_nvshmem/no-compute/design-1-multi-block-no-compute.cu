/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */
#include "../../include_nvshmem/no-compute/design-1-multi-block-no-compute.cuh"

namespace cg = cooperative_groups;

namespace Design1MultiBlockNoComputation {
__global__ void __launch_bounds__(1024, 1)
    boundary_sync_kernel(real *a_new, real *a, const int iy_start, const int iy_end, const int nx,
                         const int comm_sm_count_per_layer, const int comm_block_count_per_sm,
                         const int tile_count_x, const int iter_max,
                         uint64_t *is_done_computing_flags, const int top_iy, const int bottom_iy,
                         const int top, const int bottom, volatile int *iteration_done) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    int iter = 0;
    int cur_iter_mod = 0;
    int next_iter_mod = 1;

    const int comm_block_idx = blockIdx.x % comm_sm_count_per_layer;
    const int comm_start_block_x =
        ((comm_block_idx * comm_block_count_per_sm) % tile_count_x) * blockDim.x;
    //    const int comm_start_ix = comm_start_block_x + threadIdx.x + 1;

    while (iter < iter_max) {
        if (!grid.thread_rank()) {
            while (iteration_done[1] != iter) {
            }
        }
        cg::sync(grid);
        if (blockIdx.x < comm_sm_count_per_layer) {
            if (!cta.thread_rank()) {
                nvshmem_signal_wait_until(is_done_computing_flags +
                                              cur_iter_mod * 2 * comm_sm_count_per_layer +
                                              comm_block_idx,
                                          NVSHMEM_CMP_EQ, iter);
            }
            cg::sync(cta);

            //            int block_count = 0;
            //            int ix = comm_start_ix;

            //            for (; block_count < comm_block_count_per_sm && ix < (nx - 1); ix +=
            //            blockDim.x) {
            //                const real first_row_val =
            //                    (real(1) / real(4)) *
            //                    (a[iy_start * nx + ix + 1] + a[iy_start * nx + ix - 1] +
            //                     a[(iy_start + 1) * nx + ix] + a[(iy_start - 1) * nx + ix]);
            //                a_new[iy_start * nx + ix] = first_row_val;
            //                block_count++;
            //            }

            nvshmemx_putmem_signal_nbi_block(
                a_new + top_iy * nx + comm_start_block_x,
                a_new + iy_start * nx + comm_start_block_x,
                min(comm_block_count_per_sm * cta.num_threads(), max(0, nx - comm_start_block_x)) *
                    sizeof(real),
                is_done_computing_flags + next_iter_mod * 2 * comm_sm_count_per_layer +
                    comm_sm_count_per_layer + comm_block_idx,
                iter + 1, NVSHMEM_SIGNAL_SET, top);
        } else {
            if (!cta.thread_rank()) {
                nvshmem_signal_wait_until(is_done_computing_flags +
                                              cur_iter_mod * 2 * comm_sm_count_per_layer +
                                              comm_sm_count_per_layer + comm_block_idx,
                                          NVSHMEM_CMP_EQ, iter);
            }
            cg::sync(cta);

            //            int block_count = 0;
            //            int ix = comm_start_ix;
            //
            //            for (; block_count < comm_block_count_per_sm && ix < (nx - 1); ix +=
            //            blockDim.x) {
            //                const real last_row_val =
            //                    (real(1) / real(6)) *
            //                    (a[(iy_end - 1) * nx + ix + 1] + a[(iy_end - 1) * nx + ix - 1] +
            //                     a[iy_end * nx + ix] + a[(iy_end - 2) * nx + ix]);
            //                a_new[(iy_end - 1) * nx + ix] = last_row_val;
            //                block_count++;
            //            }

            nvshmemx_putmem_signal_nbi_block(
                a_new + bottom_iy * nx + comm_start_block_x,
                a_new + (iy_end - 1) * nx + comm_start_block_x,
                min(comm_block_count_per_sm * cta.num_threads(), max(0, nx - comm_start_block_x)) *
                    sizeof(real),
                is_done_computing_flags + next_iter_mod * 2 * comm_sm_count_per_layer +
                    comm_block_idx,
                iter + 1, NVSHMEM_SIGNAL_SET, bottom);
        }

        real *temp_pointer = a_new;
        a_new = a;
        a = temp_pointer;

        iter++;

        next_iter_mod = cur_iter_mod;
        cur_iter_mod = 1 - cur_iter_mod;

        cg::sync(grid);

        //        if (!grid.thread_rank()) {
        //            iteration_done[0] = iter;
        //        }
    }
}
}  // namespace Design1MultiBlockNoComputation

int Design1MultiBlockNoComputation::init(int argc, char *argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 512);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 512);
    const bool compare_to_single_gpu = get_arg(argv, argv + argc, "-compare");

    real *a;
    real *a_new;

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
    long long unsigned int mesh_size_per_rank = nx * ny;
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
        CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * sizeof(real)));
        CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * sizeof(real)));

        runtime_serial_non_persistent = single_gpu(nx, ny, iter_max, a_ref_h, 0, true);
    }

    nvshmem_barrier_all();

    int chunk_size;
    int chunk_size_low = (ny - 2) / npes;
    int chunk_size_high = chunk_size_low + 1;

    int num_ranks_low = npes * chunk_size_low + npes - (ny - 2);
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
    const int dim_block_y = chunk_size_high >= (num_threads_per_block / dim_block_x)
                                ? (num_threads_per_block / dim_block_x)
                                : (int)pow(2, ceil(log2(chunk_size_high)));

    const int tile_count_x = nx / (dim_block_x) + (nx % (dim_block_x) != 0);
    const int tile_count_y =
        chunk_size_high / (dim_block_y) + (chunk_size_high % (dim_block_y) != 0);

    const int comm_layer_tile_count = tile_count_x;
    const int comp_total_tile_count = tile_count_x * tile_count_y;

    const int comm_sm_count_per_layer =
        numSms / (tile_count_y + 2) + (numSms % (tile_count_y + 2) != 0);
    const int comp_sm_count = comp_total_tile_count < numSms - 2 * comm_sm_count_per_layer
                                  ? comp_total_tile_count
                                  : numSms - 2 * comm_sm_count_per_layer;

    int total_num_flags = 2 * 2 * comm_sm_count_per_layer;

    //    const int comp_block_count_per_sm =
    //        comp_total_tile_count / comp_sm_count + (comp_total_tile_count % comp_sm_count != 0);
    const int comm_block_count_per_sm = comm_layer_tile_count / comm_sm_count_per_layer +
                                        (comm_layer_tile_count % comm_sm_count_per_layer != 0);

    const int top_pe = mype > 0 ? mype - 1 : (npes - 1);
    const int bottom_pe = (mype + 1) % npes;
    int iy_end_top = (top_pe < num_ranks_low) ? chunk_size_low + 1 : chunk_size_high + 1;
    int iy_start_bottom = 0;

    nvshmem_barrier_all();

    a = (real *)nvshmem_malloc(nx * (chunk_size_high + 2) * sizeof(real));
    a_new = (real *)nvshmem_malloc(nx * (chunk_size_high + 2) * sizeof(real));

    CUDA_RT_CALL(cudaMemset((void *)a, 0, nx * (chunk_size_high + 2) * sizeof(real)));
    CUDA_RT_CALL(cudaMemset((void *)a_new, 0, nx * (chunk_size_high + 2) * sizeof(real)));

    is_done_computing_flags = (uint64_t *)nvshmem_malloc(total_num_flags * sizeof(uint64_t));
    CUDA_RT_CALL(cudaMemset(is_done_computing_flags, 0, total_num_flags * sizeof(uint64_t)));

    CUDA_RT_CALL(cudaMalloc(&iteration_done_flags, 2 * sizeof(int)));
    CUDA_RT_CALL(cudaMemset(iteration_done_flags, 0, 2 * sizeof(int)));

    // Calculate local domain boundaries
    int iy_start_global;  // My start index in the global array
    if (mype < num_ranks_low) {
        iy_start_global = mype * chunk_size_low + 1;
    } else {
        iy_start_global =
            num_ranks_low * chunk_size_low + (mype - num_ranks_low) * chunk_size_high + 1;
    }
    int iy_end_global = iy_start_global + chunk_size - 1;  // My last index in the global array

    int iy_start = 1;
    int iy_end = (iy_end_global - iy_start_global + 1) + iy_start;

    initialize_boundaries<<<(ny / npes) / 128 + 1, 128>>>(a_new, a, PI, iy_start_global - 1, nx,
                                                          chunk_size + 2, ny);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());

    dim3 comp_dim_grid(comp_sm_count);
    dim3 comp_dim_block(dim_block_x, dim_block_y);

    dim3 comm_dim_grid(comm_sm_count_per_layer * 2);
    dim3 comm_dim_block(dim_block_x * dim_block_y);

    //    void *kernelArgsInner[] = {
    //        (void *)&a_new,        (void *)&a,        (void *)&iy_start,
    //        (void *)&iy_end,       (void *)&nx,       (void *)&comp_block_count_per_sm,
    //        (void *)&tile_count_x, (void *)&iter_max, (void *)&iteration_done_flags};

    void *kernelArgsBoundary[] = {(void *)&a_new,
                                  (void *)&a,
                                  (void *)&iy_start,
                                  (void *)&iy_end,
                                  (void *)&nx,
                                  (void *)&comm_sm_count_per_layer,
                                  (void *)&comm_block_count_per_sm,
                                  (void *)&tile_count_x,
                                  (void *)&iter_max,
                                  (void *)&is_done_computing_flags,
                                  (void *)&iy_end_top,
                                  (void *)&iy_start_bottom,
                                  (void *)&top_pe,
                                  (void *)&bottom_pe,
                                  (void *)&iteration_done_flags};

    nvshmem_barrier_all();
    double start = MPI_Wtime();

    cudaStream_t inner_domain_stream;
    cudaStream_t boundary_sync_stream;

    CUDA_RT_CALL(cudaStreamCreate(&inner_domain_stream));
    CUDA_RT_CALL(cudaStreamCreate(&boundary_sync_stream));

    //    CUDA_RT_CALL(cudaLaunchCooperativeKernel(
    //        (void *)Design1MultiBlockNoComputation::jacobi_kernel, comp_dim_grid, comp_dim_block,
    //        kernelArgsInner, 0, inner_domain_stream));

    CUDA_RT_CALL((cudaError_t)nvshmemx_collective_launch(
        (void *)Design1MultiBlockNoComputation::boundary_sync_kernel, comm_dim_grid, comm_dim_block,
        kernelArgsBoundary, 0, boundary_sync_stream));

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
        CUDA_RT_CALL(cudaMemcpy(a_h + iy_start_global * nx, a + nx,
                                std::min(ny - iy_start_global, chunk_size) * nx * sizeof(real),
                                cudaMemcpyDeviceToHost));

        for (int iy = iy_start_global; result_correct && (iy <= iy_end_global); ++iy) {
            for (int ix = 1; result_correct && (ix < (nx - 1)); ++ix) {
                if (std::fabs(a_h[iy * nx + ix] - a_ref_h[iy * nx + ix]) > tol) {
                    fprintf(stderr,
                            "ERROR on rank %d: a[ %d * %d + %d] = %f does "
                            "not match %f "
                            "(reference)\n",
                            rank, iy, nx, ix, a_h[iy * nx + ix], a_ref_h[iy * nx + ix]);
                    result_correct = 0;
                }
            }
        }
    }
    int global_result_correct = 1;
    MPI_CALL(MPI_Allreduce(&result_correct, &global_result_correct, 1, MPI_INT, MPI_MIN,
                           MPI_COMM_WORLD));

    if (!mype && global_result_correct) {
        printf("Execution time: %8.4f s\n", (stop - start));

        if (compare_to_single_gpu) {
            printf(
                "Non-persistent kernel - %dx%d: 1 GPU: %8.4f s, %d GPUs: "
                "%8.4f "
                "s, speedup: "
                "%8.2f, "
                "efficiency: %8.2f \n",
                nx, ny, runtime_serial_non_persistent, npes, (stop - start),
                runtime_serial_non_persistent / (stop - start),
                runtime_serial_non_persistent / (npes * (stop - start)) * 100);
        }
    }
    nvshmem_free((void *)a);
    nvshmem_free((void *)a_new);
    nvshmem_free(is_done_computing_flags);

    if (compare_to_single_gpu) {
        CUDA_RT_CALL(cudaFreeHost(a_h));
        CUDA_RT_CALL(cudaFreeHost(a_ref_h));
    }

    nvshmem_finalize();
    MPI_CALL(MPI_Finalize());
    return (result_correct == 1) ? 0 : 1;
}
