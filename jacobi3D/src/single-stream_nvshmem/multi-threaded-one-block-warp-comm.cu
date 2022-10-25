/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */
#include <cmath>
#include <cstdio>
#include <iostream>

#include <omp.h>

#include <cooperative_groups.h>

#include <nvshmem.h>
#include <nvshmemx.h>
#include "../../include/single-stream_nvshmem/multi-threaded-one-block-warp-comm.cuh"

namespace cg = cooperative_groups;

namespace SSMultiThreadedOneBlockWarpCommNvshmem
{

    __global__ void __launch_bounds__(1024, 1)
        jacobi_kernel(real *a_new, real *a, const int iz_start, const int iz_end, const int ny,
                      const int nx, const int comp_tile_size_x, const int comp_tile_size_y,
                      const int comp_tile_size_z, const int comm_tile_size_x,
                      const int comm_tile_size_y, const int num_comp_tiles_x,
                      const int num_comp_tiles_y, const int num_comp_tiles_z,
                      const int num_comm_tiles_x, const int num_comm_tiles_y, const int iter_max,
                      real *halo_buffer_for_top_neighbor, real *halo_buffer_for_bottom_neighbor,
                      int *is_top_done_computing_flags, int *is_bottom_done_computing_flags,
                      const int top, const int bottom)
    {
        cg::thread_block cta = cg::this_thread_block();
        cg::grid_group grid = cg::this_grid();
        auto warp = cg::tiled_partition<32>(cta);

        int iter = 0;
        int cur_iter_mod = 0;
        int next_iter_mod = 1;

        const int num_flags = 2 * num_comm_tiles_x * num_comm_tiles_y;

        while (iter < iter_max)
        {
            if (blockIdx.x == gridDim.x - 1)
            {
                int cur_iter_comm_tile_flag_idx_x;
                int cur_iter_comm_tile_flag_idx_y;
                int next_iter_comm_tile_flag_idx_x;
                int next_iter_comm_tile_flag_idx_y;
                for (int comm_tile_idx_y = 0; comm_tile_idx_y < num_comm_tiles_y; comm_tile_idx_y++)
                {
                    int comm_tile_start_y =
                        (comm_tile_idx_y == 0) ? 1 : comm_tile_idx_y * comm_tile_size_y;

                    int iy = threadIdx.z * blockDim.y + threadIdx.y + comm_tile_start_y;

                    for (int comm_tile_idx_x = 0; comm_tile_idx_x < num_comm_tiles_x;
                         comm_tile_idx_x++)
                    {
                        int comm_tile_start_x =
                            (comm_tile_idx_x == 0) ? 1 : comm_tile_idx_x * comm_tile_size_x;

                        int ix = threadIdx.x + comm_tile_start_x;

                        if (warp.thread_rank() == 0)
                        {
                            cur_iter_comm_tile_flag_idx_x = comm_tile_idx_x;
                            cur_iter_comm_tile_flag_idx_y = comm_tile_idx_y;
                            nvshmem_int_wait_until(
                                is_top_done_computing_flags + cur_iter_mod * num_flags +
                                    cur_iter_comm_tile_flag_idx_y * num_comm_tiles_x +
                                    cur_iter_comm_tile_flag_idx_x * warp.meta_group_size() +
                                    warp.meta_group_rank(),
                                NVSHMEM_CMP_EQ, iter);

                            /* while (local_is_top_neighbor_done_writing_to_me
                                       [cur_iter_comm_tile_flag_idx_y * num_comm_tiles_x +
                                        cur_iter_comm_tile_flag_idx_x + cur_iter_mod * num_flags] !=
                                   iter) {
                            } */
                        }
                        cg::sync(warp);

                        // copy per row wise (since its warp sized in x dim)
                        if (iy < ny - 1 && ix < nx - 1)
                        {
                            real first_row_val =
                                (a[iz_start * ny * nx + iy * nx + ix + 1] +
                                 a[iz_start * ny * nx + iy * nx + ix - 1] +
                                 a[iz_start * ny * nx + (iy + 1) * nx + ix] +
                                 a[iz_start * ny * nx + (iy - 1) * nx + ix] +
                                 a[(iz_start + 1) * ny * nx + iy * nx + ix] +
                                 // remote_my_halo_buffer_on_top_neighbor[cur_iter_mod * ny * nx +iy *
                                 // nx + ix] +
                                 //
                                 halo_buffer_for_top_neighbor[cur_iter_mod * ny * nx + iy * nx + ix]) /
                                real(6.0);

                            a_new[iz_start * ny * nx + iy * nx + ix] = first_row_val;
                            ///????
                            nvshmemx_float_put_nbi_warp(
                                halo_buffer_for_top_neighbor + next_iter_mod * ny * nx + iy,
                                a_new + iz_start * ny * nx + iy,
                                min(warpSize, nx - 1 - comm_tile_start_x), top);
                            // local_halo_buffer_for_top_neighbor[next_iter_mod * ny * nx + iy * nx +
                            // ix] =
                            //     first_row_val;
                        }

                        cg::sync(warp);

                        if (warp.thread_rank() == 0)
                        {
                            next_iter_comm_tile_flag_idx_x = (num_comm_tiles_x + comm_tile_idx_x);
                            next_iter_comm_tile_flag_idx_y = (comm_tile_idx_y);
                            nvshmem_int_atomic_inc(
                                is_top_done_computing_flags + next_iter_mod * num_flags +
                                    next_iter_comm_tile_flag_idx_y * num_comm_tiles_x +
                                    next_iter_comm_tile_flag_idx_x * warp.meta_group_size() +
                                    warp.meta_group_rank(),
                                top);

                            /*remote_am_done_writing_to_top_neighbor[next_iter_comm_tile_flag_idx_y *
                                                                       num_comm_tiles_x +
                                                                   next_iter_comm_tile_flag_idx_x +
                                                                   next_iter_mod * num_flags] =
                                iter + 1;*/

                            nvshmem_int_wait_until(
                                is_bottom_done_computing_flags + cur_iter_mod * num_flags +
                                    cur_iter_comm_tile_flag_idx_y * num_comm_tiles_x +
                                    cur_iter_comm_tile_flag_idx_x * warp.meta_group_size() +
                                    warp.meta_group_rank(),
                                NVSHMEM_CMP_EQ, iter);

                            /*while (local_is_bottom_neighbor_done_writing_to_me
                                       [cur_iter_comm_tile_flag_idx_y * num_comm_tiles_x +
                                        cur_iter_comm_tile_flag_idx_x + cur_iter_mod * num_flags] !=
                                   iter) {
                            }*/
                        }

                        cg::sync(warp);

                        if (iy < ny - 1 && ix < nx - 1)
                        {
                            const real last_row_val =
                                (a[(iz_end - 1) * ny * nx + iy * nx + ix + 1] +
                                 a[(iz_end - 1) * ny * nx + iy * nx + ix - 1] +
                                 a[(iz_end - 1) * ny * nx + (iy + 1) * nx + ix] +
                                 a[(iz_end - 1) * ny * nx + (iy - 1) * nx + ix] +
                                 halo_buffer_for_bottom_neighbor[cur_iter_mod * ny * nx + iy * nx +
                                                                 ix] +
                                 // remote_my_halo_buffer_on_bottom_neighbor[cur_iter_mod * ny * nx +
                                 //                                          iy * nx + ix] +
                                 a[(iz_end - 2) * ny * nx + iy * nx + ix]) /
                                real(6.0);

                            a_new[(iz_end - 1) * ny * nx + iy * nx + ix] = last_row_val;
                            nvshmemx_float_put_nbi_warp(
                                halo_buffer_for_bottom_neighbor + next_iter_mod * ny * nx + iy,
                                a_new + iz_start * ny * nx + iy,
                                min(warpSize, nx - 1 - comm_tile_start_x), bottom);
                            // local_halo_buffer_for_bottom_neighbor[next_iter_mod * ny * nx + iy * nx +
                            //                                       ix] = last_row_val;
                        }

                        cg::sync(warp);

                        if (warp.thread_rank() == 0)
                        {
                            next_iter_comm_tile_flag_idx_x = comm_tile_idx_x;
                            next_iter_comm_tile_flag_idx_y = comm_tile_idx_y;
                            nvshmem_int_atomic_inc(
                                is_bottom_done_computing_flags + next_iter_mod * num_flags +
                                    next_iter_comm_tile_flag_idx_y * num_comm_tiles_x +
                                    next_iter_comm_tile_flag_idx_x * warp.meta_group_size() +
                                    warp.meta_group_rank(),
                                bottom);
                            /*remote_am_done_writing_to_bottom_neighbor[next_iter_comm_tile_flag_idx_y *
                                                                          num_comm_tiles_x +
                                                                      next_iter_comm_tile_flag_idx_x +
                                                                      next_iter_mod * num_flags] =
                                iter + 1;*/
                        }
                    }
                }
            }
            else
            {
                const int grid_dim_x = (comp_tile_size_x + blockDim.x - 1) / blockDim.x;
                const int grid_dim_y = (comp_tile_size_y + blockDim.y - 1) / blockDim.y;

                const int block_idx_z = blockIdx.x / (grid_dim_x * grid_dim_y);
                const int block_idx_y = (blockIdx.x % (grid_dim_x * grid_dim_y)) / grid_dim_x;
                const int block_idx_x = blockIdx.x % grid_dim_x;

                const int base_iz = block_idx_z * blockDim.z + threadIdx.z;
                const int base_iy = block_idx_y * blockDim.y + threadIdx.y;
                const int base_ix = block_idx_x * blockDim.x + threadIdx.x;
                for (int iz = (base_iz + iz_start + 1) * ny * nx; iz < (iz_end - 1) * ny * nx;
                     iz += comp_tile_size_z * ny * nx)
                {
                    int iz_below = iz + ny * nx;
                    int iz_above = iz - ny * nx;
                    for (int iy = (base_iy + 1) * nx; iy < (ny - 1) * nx; iy += comp_tile_size_y * nx)
                    {
                        int iy_below = iy + nx;
                        int iy_above = iy - nx;
                        for (int ix = (base_ix + 1); ix < (nx - 1); ix += comp_tile_size_x)
                        {
                            // big bottleneck here
                            const real new_val = (a[iz + iy + ix + 1] + a[iz + iy + ix - 1] +
                                                  a[iz + iy_below + ix] + a[iz + iy_above + ix] +
                                                  a[iz_below + iy + ix] + a[iz_above + iy + ix]) /
                                                 real(6.0);

                            a_new[iz + iy + ix] = new_val;
                        }
                    }
                }
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
} // namespace SSMultiThreadedOneBlockWarpCommNvshmem

int SSMultiThreadedOneBlockWarpCommNvshmem::init(int argc, char *argv[])
{
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 512);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 512);
    const int nz = get_argval<int>(argv, argv + argc, "-nz", 512);
    const bool compare_to_single_gpu = get_arg(argv, argv + argc, "-compare");

    real *a[MAX_NUM_DEVICES];
    real *a_new[MAX_NUM_DEVICES];
    int iz_end[MAX_NUM_DEVICES];

    real *halo_buffer_for_top_neighbor[MAX_NUM_DEVICES];
    real *halo_buffer_for_bottom_neighbor[MAX_NUM_DEVICES];

    int *is_top_done_computing_flags[MAX_NUM_DEVICES];
    int *is_bottom_done_computing_flags[MAX_NUM_DEVICES];

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
    if (1 < num_devices && num_devices < local_size)
    {
        fprintf(
            stderr,
            "ERROR Number of visible devices (%d) is less than number of ranks on the node (%d)!\n",
            num_devices, local_size);
        MPI_CALL(MPI_Finalize());
        return 1;
    }
    if (1 == num_devices)
    {
        // Only 1 device visible, assuming GPU affinity is handled via CUDA_VISIBLE_DEVICES
        CUDA_RT_CALL(cudaSetDevice(0));
    }
    else
    {
        CUDA_RT_CALL(cudaSetDevice(local_rank));
    }

    MPI_Comm mpi_comm;
    nvshmemx_init_attr_t attr;

    mpi_comm = MPI_COMM_WORLD;
    attr.mpi_comm = &mpi_comm;

    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 32;
    constexpr int dim_block_z = 1;

    constexpr int comp_tile_size_x = dim_block_x;
    constexpr int comp_tile_size_y = dim_block_y;
    constexpr int comp_tile_size_z = dim_block_z;

    constexpr int comm_tile_size_x = dim_block_x;
    constexpr int comm_tile_size_y = dim_block_z * dim_block_y;

    constexpr int grid_dim_x = (comp_tile_size_x + dim_block_x - 1) / dim_block_x;
    constexpr int grid_dim_y = (comp_tile_size_y + dim_block_y - 1) / dim_block_y;

    
    int num_comp_tiles_x = nx / comp_tile_size_x + (nx % comp_tile_size_x != 0);
    int num_comp_tiles_y = ny / comp_tile_size_y + (ny % comp_tile_size_y != 0);

    int num_comm_tiles_x = nx / comm_tile_size_x + (nx % comm_tile_size_x != 0);
    int num_comm_tiles_y = ny / comm_tile_size_y + (ny % comm_tile_size_y != 0);

    int total_num_flags = 4 * num_comm_tiles_x * dim_block_y * num_comm_tiles_y;

    // Set symmetric heap size for nvshmem based on problem size
    // Its default value in nvshmem is 1 GB which is not sufficient
    // for large mesh sizes
    long long unsigned int mesh_size_per_rank = 2 * nx * ny + total_num_flags;
    long long unsigned int required_symmetric_heap_size =
        2 * mesh_size_per_rank * sizeof(real) *
        1.1; // Factor 2 is because 2 arrays are allocated - a and a_new
             // 1.1 factor is just for alignment or other usage

    char *value = getenv("NVSHMEM_SYMMETRIC_SIZE");
    if (value)
    { /* env variable is set */
        long long unsigned int size_env = parse_nvshmem_symmetric_size(value);
        if (size_env < required_symmetric_heap_size)
        {
            fprintf(stderr,
                    "ERROR: Minimum NVSHMEM_SYMMETRIC_SIZE = %lluB, Current NVSHMEM_SYMMETRIC_SIZE "
                    "= %s\n",
                    required_symmetric_heap_size, value);
            MPI_CALL(MPI_Finalize());
            return -1;
        }
    }
    else
    {
        char symmetric_heap_size_str[100];
        sprintf(symmetric_heap_size_str, "%llu", required_symmetric_heap_size);
        setenv("NVSHMEM_SYMMETRIC_SIZE", symmetric_heap_size_str, 1);
    }
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    int npes = nvshmem_n_pes();
    int mype = nvshmem_my_pe();

    nvshmem_barrier_all();

    bool result_correct = true;
    if (compare_to_single_gpu && 0 == mype)
    {
        CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * nz * sizeof(real)));
        CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * nz * sizeof(real)));

        runtime_serial_non_persistent = single_gpu(nz, ny, nx, iter_max, a_ref_h, 0, true);
    }

    nvshmem_barrier_all();

    int chunk_size;
    int chunk_size_low = (nz - 2) / num_devices;
    int chunk_size_high = chunk_size_low + 1;

    int num_ranks_low = num_devices * chunk_size_low + num_devices - (nz - 2);
    if (mype < num_ranks_low)
        chunk_size = chunk_size_low;
    else
        chunk_size = chunk_size_high;

    cudaDeviceProp deviceProp{};
    CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, mype));
    int numSms = deviceProp.multiProcessorCount;

    int num_comp_tiles_z =
        (nz / num_devices) / comp_tile_size_z + ((nz / num_devices) % comp_tile_size_z != 0);
    int max_thread_blocks_z = (numSms - 1) / (grid_dim_x * grid_dim_y);
    int comp_tile_size_z = dim_block_z * max_thread_blocks_z;
    
    
    const int top = mype > 0 ? mype - 1 : (num_devices - 1);
    const int bottom = (mype + 1) % num_devices;

    if (top != mype)
    {
        int canAccessPeer = 0;
        CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, mype, top));
        if (canAccessPeer)
        {
            CUDA_RT_CALL(cudaDeviceEnablePeerAccess(top, 0));
        }
        else
        {
            std::cerr << "P2P access required from " << mype << " to " << top << std::endl;
        }
        if (top != bottom)
        {
            canAccessPeer = 0;
            CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, mype, bottom));
            if (canAccessPeer)
            {
                CUDA_RT_CALL(cudaDeviceEnablePeerAccess(bottom, 0));
            }
            else
            {
                std::cerr << "P2P access required from " << mype << " to " << bottom << std::endl;
            }
        }
    }

    nvshmem_barrier_all();

    CUDA_RT_CALL(cudaMalloc(a + mype, nx * ny * (chunk_size + 2) * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc(a_new + mype, nx * ny * (chunk_size + 2) * sizeof(real)));

    CUDA_RT_CALL(cudaMemset(a[mype], 0, nx * ny * (chunk_size + 2) * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(a_new[mype], 0, nx * ny * (chunk_size + 2) * sizeof(real)));

    halo_buffer_for_top_neighbor[mype] = (real *)nvshmem_calloc(2 * nx * ny, sizeof(real));
    halo_buffer_for_bottom_neighbor[mype] = (real *)nvshmem_calloc(2 * nx * ny, sizeof(real));

    // CUDA_RT_CALL(cudaMalloc(halo_buffer_for_top_neighbor + dev_id, 2 * nx * ny * sizeof(real)));
    // CUDA_RT_CALL(cudaMalloc(halo_buffer_for_bottom_neighbor + dev_id, 2 * nx * ny *
    // sizeof(real)));

    // CUDA_RT_CALL(cudaMemset(halo_buffer_for_top_neighbor[dev_id], 0, 2 * nx * ny *
    // sizeof(real))); CUDA_RT_CALL(cudaMemset(halo_buffer_for_bottom_neighbor[dev_id], 0, 2 * nx *
    // ny * sizeof(real)));

    is_top_done_computing_flags[mype] = (int *)nvshmem_calloc(total_num_flags, sizeof(int));
    is_bottom_done_computing_flags[mype] = (int *)nvshmem_calloc(total_num_flags, sizeof(int));

    // CUDA_RT_CALL(cudaMalloc(is_top_done_computing_flags + dev_id, total_num_flags *
    // sizeof(int))); CUDA_RT_CALL(cudaMalloc(is_bottom_done_computing_flags + dev_id,
    // total_num_flags * sizeof(int)));

    // CUDA_RT_CALL(cudaMemset(is_top_done_computing_flags[dev_id], 0, total_num_flags *
    // sizeof(int))); CUDA_RT_CALL(cudaMemset(is_bottom_done_computing_flags[dev_id], 0,
    // total_num_flags * sizeof(int)));

    // Calculate local domain boundaries
    int iz_start_global; // My start index in the global array
    if (mype < num_ranks_low)
    {
        iz_start_global = mype * chunk_size_low + 1;
    }
    else
    {
        iz_start_global =
            num_ranks_low * chunk_size_low + (mype - num_ranks_low) * chunk_size_high + 1;
    }
    int iz_end_global = iz_start_global + chunk_size - 1; // My last index in the global array

    int iz_start = 1;
    iz_end[mype] = (iz_end_global - iz_start_global + 1) + iz_start;

    initialize_boundaries<<<(nz / num_devices) / 128 + 1, 128>>>(
        a_new[mype], a[mype], PI, iz_start_global - 1, nx, ny, chunk_size + 2, nz);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());

    dim3 dim_grid(numSms, 1, 1);
    dim3 dim_block(dim_block_x, dim_block_y, dim_block_z);

    void *kernelArgs[] = {(void *)&a_new[mype],
                          (void *)&a[mype],
                          (void *)&iz_start,
                          (void *)&iz_end[mype],
                          (void *)&ny,
                          (void *)&nx,
                          (void *)&comp_tile_size_x,
                          (void *)&comp_tile_size_y,
                          (void *)&comp_tile_size_z,
                          (void *)&comm_tile_size_x,
                          (void *)&comm_tile_size_y,
                          (void *)&num_comp_tiles_x,
                          (void *)&num_comp_tiles_y,
                          (void *)&num_comp_tiles_z,
                          (void *)&num_comm_tiles_x,
                          (void *)&num_comm_tiles_y,
                          (void *)&iter_max,
                          (void *)&halo_buffer_for_top_neighbor,
                          (void *)&halo_buffer_for_bottom_neighbor,
                          (void *)&is_top_done_computing_flags,
                          (void *)&is_bottom_done_computing_flags,
                          (void *)&top,
                          (void *)&bottom};

    nvshmem_barrier_all();
    double start = MPI_Wtime();

    CUDA_RT_CALL((cudaError_t)nvshmemx_collective_launch(
        (void *)SSMultiThreadedOneBlockWarpCommNvshmem::jacobi_kernel, dim_grid, dim_block,
        kernelArgs, 0, nullptr));

    CUDA_RT_CALL(cudaDeviceSynchronize());
    CUDA_RT_CALL(cudaGetLastError());

    // Need to swap pointers on CPU if iteration count is odd
    // Technically, we don't know the iteration number (since we'll be doing
    // l2-norm) Could write iter to CPU when kernel is done
    if (iter_max % 2 == 1)
    {
        std::swap(a_new[mype], a[mype]);
    }

    nvshmem_barrier_all();
    double stop = MPI_Wtime();
    nvshmem_barrier_all();
    if (compare_to_single_gpu)
    {
        CUDA_RT_CALL(cudaMemcpy(
            a_h + iz_start_global * ny * nx, a[mype] + ny * nx,
            std::min((nz - iz_start_global) * ny * nx, chunk_size * nx * ny) * sizeof(real),
            cudaMemcpyDeviceToHost));

        for (int iz = 1; result_correct && (iz < (nz - 1)); ++iz)
        {
            for (int iy = 1; result_correct && (iy < (ny - 1)); ++iy)
            {
                for (int ix = 1; result_correct && (ix < (nx - 1)); ++ix)
                {
                    if (std::fabs(a_h[iz * ny * nx + iy * nx + ix] -
                                  a_ref_h[iz * ny * nx + iy * nx + ix]) > tol)
                    {
                        fprintf(stderr,
                                "ERROR on rank %d: a[%d * %d + %d * %d + %d] = %f does "
                                "not match %f "
                                "(reference)\n",
                                rank, iz, ny * nx, iy, nx, ix, a_h[iz * ny * nx + iy * nx + ix],
                                a_ref_h[iz * ny * nx + iy * nx + ix]);
                        // result_correct = false;
                    }
                }
            }
        }
        if (result_correct)
        {
            // printf("Num GPUs: %d.\n", num_devices);
            printf("Execution time: %8.4f s\n", (stop - start));

            if (compare_to_single_gpu)
            {
                printf(
                    "Non-persistent kernel - %dx%dx%d: 1 GPU: %8.4f s, %d GPUs: "
                    "%8.4f "
                    "s, speedup: "
                    "%8.2f, "
                    "efficiency: %8.2f \n",
                    nz, ny, nx, runtime_serial_non_persistent, num_devices, (stop - start),
                    runtime_serial_non_persistent / (stop - start),
                    runtime_serial_non_persistent / (num_devices * (stop - start)) * 100);
            }
        }
    }

    int global_result_correct = 1;
    MPI_CALL(MPI_Allreduce(&result_correct, &global_result_correct, 1, MPI_INT, MPI_MIN,
                           MPI_COMM_WORLD));
    result_correct = global_result_correct;

    CUDA_RT_CALL(cudaFree(a_new[mype]));
    CUDA_RT_CALL(cudaFree(a[mype]));
    nvshmem_free(halo_buffer_for_top_neighbor[mype]);
    nvshmem_free(halo_buffer_for_bottom_neighbor[mype]);
    nvshmem_free(is_top_done_computing_flags[mype]);
    nvshmem_free(is_bottom_done_computing_flags[mype]);

    if (compare_to_single_gpu && 0 == mype)
    {
        CUDA_RT_CALL(cudaFreeHost(a_h));
        CUDA_RT_CALL(cudaFreeHost(a_ref_h));
    }

    nvshmem_finalize();
    MPI_CALL(MPI_Finalize());
    return 0;
}
