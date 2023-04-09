/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */
#include <cmath>
#include <iostream>

#include <cuda_runtime.h>

#include <omp.h>

#include <cooperative_groups.h>

#define HALO (1)

// perks stuff
// DON'T CHANGE THE ORDER
#include "./common/cuda_common.cuh"
#include "./common/cuda_computation.cuh"
#include "config.cuh"
// #include "./common/jacobi_cuda.cuh"
// #include "./common/jacobi_reference.hpp"
// #include "./common/types.hpp"
#include "./perksconfig.cuh"

#include "../../include/PERKS/multi-stream-perks.cuh"
#include "../../include/common.h"

namespace cg = cooperative_groups;

namespace MultiStreamPERKS {
__global__ void __launch_bounds__(1024, 1)
    boundary_sync_kernel(real *a_new, real *a, const int iz_start, const int iz_end, const int ny,
                         const int nx, const int comm_tile_size_x, const int comm_tile_size_y,
                         const int num_comm_tiles_x, const int num_comm_tiles_y, const int iter_max,
                         volatile real *local_halo_buffer_for_top_neighbor,
                         volatile real *local_halo_buffer_for_bottom_neighbor,
                         volatile real *remote_my_halo_buffer_on_top_neighbor,
                         volatile real *remote_my_halo_buffer_on_bottom_neighbor,
                         volatile int *local_is_top_neighbor_done_writing_to_me,
                         volatile int *local_is_bottom_neighbor_done_writing_to_me,
                         volatile int *remote_am_done_writing_to_top_neighbor,
                         volatile int *remote_am_done_writing_to_bottom_neighbor,
                         volatile int *iteration_done) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    int num_flags = 2 * num_comm_tiles_x * num_comm_tiles_y;

    int iter = 0;

    int cur_iter_mod = 0;
    int next_iter_mod = 1;
    int temp_iter_mod = 0;

    int comm_tile_idx_y;
    int comm_tile_start_y;

    int comm_tile_idx_x;
    int comm_tile_start_x;

    int cur_iter_comm_tile_flag_idx_x;
    int cur_iter_comm_tile_flag_idx_y;
    int next_iter_comm_tile_flag_idx_x;
    int next_iter_comm_tile_flag_idx_y;

    int iy = 0;
    int ix = 0;

    while (iter < iter_max) {
        while (iteration_done[1] != iter) {
        }

        if (blockIdx.x == gridDim.x - 1) {
            for (comm_tile_idx_y = 0; comm_tile_idx_y < num_comm_tiles_y; comm_tile_idx_y++) {
                comm_tile_start_y = (comm_tile_idx_y == 0) ? 1 : comm_tile_idx_y * comm_tile_size_y;

                iy = threadIdx.z * blockDim.y + threadIdx.y + comm_tile_start_y;

                for (comm_tile_idx_x = 0; comm_tile_idx_x < num_comm_tiles_x; comm_tile_idx_x++) {
                    comm_tile_start_x =
                        (comm_tile_idx_x == 0) ? 1 : comm_tile_idx_x * comm_tile_size_x;

                    ix = threadIdx.x + comm_tile_start_x;

                    if (cta.thread_rank() == 0) {
                        cur_iter_comm_tile_flag_idx_x = comm_tile_idx_x;
                        cur_iter_comm_tile_flag_idx_y = comm_tile_idx_y;

                        while (local_is_top_neighbor_done_writing_to_me
                                   [cur_iter_comm_tile_flag_idx_y * num_comm_tiles_x +
                                    cur_iter_comm_tile_flag_idx_x + cur_iter_mod * num_flags] !=
                               iter) {
                        }
                    }

                    cg::sync(cta);

                    if (iy < ny - 1 && ix < nx - 1) {
                        const real top =
                            remote_my_halo_buffer_on_top_neighbor[cur_iter_mod * ny * nx + iy * nx +
                                                                  ix];
                        // https://github.com/neozhang307/PERKS/blob/master/stencil/3dstencil/3d7pt/3d7pt_gold.cpp
                        const real first_row_val =
                            0.161f * a[iz_start * ny * nx + iy * nx + ix + 1]      // east
                            + 0.162f * a[iz_start * ny * nx + iy * nx + ix - 1]    // west
                            + 0.163f * a[iz_start * ny * nx + (iy + 1) * nx + ix]  // north
                            + 0.164f * a[iz_start * ny * nx + (iy - 1) * nx + ix]  // south
                            + 0.165f * top                                         // top
                            + 0.166f * a[(iz_start - 1) * ny * nx + iy * nx + ix]  // bottom
                            - 1.67f * a[iz_start * ny * nx + iy * nx + ix];        // center

                        a_new[iz_start * ny * nx + iy * nx + ix] = first_row_val;
                        local_halo_buffer_for_top_neighbor[next_iter_mod * ny * nx + iy * nx + ix] =
                            first_row_val;
                    }

                    cg::sync(cta);

                    if (cta.thread_rank() == 0) {
                        next_iter_comm_tile_flag_idx_x = (num_comm_tiles_x + comm_tile_idx_x);
                        next_iter_comm_tile_flag_idx_y = (comm_tile_idx_y);

                        remote_am_done_writing_to_top_neighbor[next_iter_comm_tile_flag_idx_y *
                                                                   num_comm_tiles_x +
                                                               next_iter_comm_tile_flag_idx_x +
                                                               next_iter_mod * num_flags] =
                            iter + 1;
                    }
                }
            }
        } else if (blockIdx.x == gridDim.x - 2) {
            for (comm_tile_idx_y = 0; comm_tile_idx_y < num_comm_tiles_y; comm_tile_idx_y++) {
                comm_tile_start_y = (comm_tile_idx_y == 0) ? 1 : comm_tile_idx_y * comm_tile_size_y;
                iy = threadIdx.z * blockDim.y + threadIdx.y + comm_tile_start_y;

                for (comm_tile_idx_x = 0; comm_tile_idx_x < num_comm_tiles_x; comm_tile_idx_x++) {
                    comm_tile_start_x =
                        (comm_tile_idx_x == 0) ? 1 : comm_tile_idx_x * comm_tile_size_x;
                    ix = threadIdx.x + comm_tile_start_x;

                    if (cta.thread_rank() == 0) {
                        cur_iter_comm_tile_flag_idx_x = (num_comm_tiles_x + comm_tile_idx_x);
                        cur_iter_comm_tile_flag_idx_y = (comm_tile_idx_y);
                        while (local_is_bottom_neighbor_done_writing_to_me
                                   [cur_iter_comm_tile_flag_idx_y * num_comm_tiles_x +
                                    cur_iter_comm_tile_flag_idx_x + cur_iter_mod * num_flags] !=
                               iter) {
                        }
                    }

                    cg::sync(cta);

                    if (iy < ny - 1 && ix < nx - 1) {
                        const real bottom =
                            remote_my_halo_buffer_on_bottom_neighbor[cur_iter_mod * ny * nx +
                                                                     iy * nx + ix];
                        const real last_row_val =
                            0.161f * a[(iz_end - 1) * ny * nx + iy * nx + ix + 1]      // east
                            + 0.162f * a[(iz_end - 1) * ny * nx + iy * nx + ix - 1]    // west
                            + 0.163f * a[(iz_end - 1) * ny * nx + (iy + 1) * nx + ix]  // north
                            + 0.164f * a[(iz_end - 1) * ny * nx + (iy - 1) * nx + ix]  // south
                            + 0.166f * bottom                                          // bottom
                            + 0.165f * a[(iz_end - 2) * ny * nx + iy * nx + ix]        // what?
                            - 1.67f * a[(iz_end - 1) * ny * nx + (iy - 1) * nx + ix];

                        a_new[(iz_end - 1) * ny * nx + iy * nx + ix] = last_row_val;
                        local_halo_buffer_for_bottom_neighbor[next_iter_mod * ny * nx + iy * nx +
                                                              ix] = last_row_val;
                    }

                    cg::sync(cta);

                    if (cta.thread_rank() == 0) {
                        next_iter_comm_tile_flag_idx_x = comm_tile_idx_x;
                        next_iter_comm_tile_flag_idx_y = comm_tile_idx_y;

                        remote_am_done_writing_to_bottom_neighbor[next_iter_comm_tile_flag_idx_y *
                                                                      num_comm_tiles_x +
                                                                  next_iter_comm_tile_flag_idx_x +
                                                                  next_iter_mod * num_flags] =
                            iter + 1;
                    }
                }
            }
        }

        real *temp_pointer_first = a_new;
        a_new = a;
        a = temp_pointer_first;

        iter++;

        temp_iter_mod = cur_iter_mod;
        cur_iter_mod = next_iter_mod;
        next_iter_mod = temp_iter_mod;

        cg::sync(grid);

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            iteration_done[0] = iter;
        }

        cg::sync(grid);
    }
}
}  // namespace MultiStreamPERKS

// #pragma clang diagnostic push
// #pragma ide diagnostic ignored "openmp-use-default-none"
int MultiStreamPERKS::init(int argc, char *argv[]) {
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

    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));
    //    num_devices = 1;

    // ------------------------------------
    // PERKS config
    // ------------------------------------

    // Buffers
    //    real(*input_h)[nx] = (real(*)[nx])getRandom2DArray(ny, nx);
    //    real(*output)[nx] = (real(*)[nx])getZero2DArray(ny, nx);
    //    real(*output_gold)[nx] = (real(*)[nx])getZero2DArray(ny, nx);

#define ITEM_PER_THREAD (8)
#define REG_FOLDER_Z (0)
    // #define TILE_X 256

    // damnit
    //    if (blkpsm <= 0) blkpsm = 100;

    bool useSM = true;
    bool isDoubleTile = true;

    // Change this later
    int ptx = 800;

    // 128 or 256
    int bdimx = 256;
    int blkpsm = 100;

    bdimx = bdimx == 128 ? 128 : 256;
    if (isDoubleTile) {
        if (bdimx == 256) blkpsm = 1;
        if (bdimx == 128) blkpsm = min(blkpsm, 2);
    }

    //    const int LOCAL_ITEM_PER_THREAD = isDoubleTile ? ITEM_PER_THREAD * 2 : ITEM_PER_THREAD;

#undef __PRINT__
#define PERSISTENTLAUNCH

#define REAL real

//    num_devices = 1;
#pragma omp parallel num_threads(num_devices)
    {
        int dev_id = omp_get_thread_num();

        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaFree(nullptr));

        if (compare_to_single_gpu && 0 == dev_id) {
            CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * nz * sizeof(real)));
            CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * nz * sizeof(real)));

            runtime_serial_non_persistent = single_gpu(nz, ny, nx, iter_max, a_ref_h, 0, true);
        }

#pragma omp barrier

        int chunk_size;
        int chunk_size_low = (nz - 2) / num_devices;
        int chunk_size_high = chunk_size_low + 1;

        cudaDeviceProp deviceProp{};
        CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, dev_id));
        int numSms = deviceProp.multiProcessorCount;

        constexpr int dim_block_x = 32;
        constexpr int dim_block_y = 32;
        constexpr int dim_block_z = 1;

        //        constexpr int comp_tile_size_x = dim_block_x;
        //        constexpr int comp_tile_size_y = dim_block_y;

        constexpr int comm_tile_size_x = dim_block_x;
        constexpr int comm_tile_size_y = dim_block_z * dim_block_y;

        //        constexpr int grid_dim_x = (comp_tile_size_x + dim_block_x - 1) / dim_block_x;
        //        constexpr int grid_dim_y = (comp_tile_size_y + dim_block_y - 1) / dim_block_y;

        //        int max_thread_blocks_z = (numSms - 2) / (grid_dim_x * grid_dim_y);

        int num_comm_tiles_x = nx / comm_tile_size_x + (nx % comm_tile_size_x != 0);
        int num_comm_tiles_y = ny / comm_tile_size_y + (ny % comm_tile_size_y != 0);

        int total_num_flags = 4 * comm_tile_size_x * comm_tile_size_y;

        int num_ranks_low = num_devices * chunk_size_low + num_devices - (nz - 2);
        if (dev_id < num_ranks_low)
            chunk_size = chunk_size_low;
        else
            chunk_size = chunk_size_high;

        const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);
        const int bottom = (dev_id + 1) % num_devices;

        if (top != dev_id) {
            int canAccessPeer = 0;
            CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, top));
            if (canAccessPeer) {
                CUDA_RT_CALL(cudaDeviceEnablePeerAccess(top, 0));
            } else {
                std::cerr << "P2P access required from " << dev_id << " to " << top << std::endl;
            }
            if (top != bottom) {
                canAccessPeer = 0;
                CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, bottom));
                if (canAccessPeer) {
                    CUDA_RT_CALL(cudaDeviceEnablePeerAccess(bottom, 0));
                } else {
                    std::cerr << "P2P access required from " << dev_id << " to " << bottom
                              << std::endl;
                }
            }
        }

#pragma omp barrier

        int *iteration_done_flags[2];

        CUDA_RT_CALL(cudaMalloc(iteration_done_flags, 2 * sizeof(int)));
        CUDA_RT_CALL(cudaMalloc(iteration_done_flags, 2 * sizeof(int)));

        CUDA_RT_CALL(cudaMalloc(iteration_done_flags + 1, 2 * sizeof(int)));
        CUDA_RT_CALL(cudaMalloc(iteration_done_flags + 1, 2 * sizeof(int)));

        CUDA_RT_CALL(cudaMemset(iteration_done_flags[0], 0, 2 * sizeof(int)));
        CUDA_RT_CALL(cudaMemset(iteration_done_flags[1], 0, 2 * sizeof(int)));

        CUDA_RT_CALL(cudaMalloc(a + dev_id, nx * ny * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(a_new + dev_id, nx * ny * (chunk_size + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMemset(a[dev_id], 0, nx * ny * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(a_new[dev_id], 0, nx * ny * (chunk_size + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMalloc(halo_buffer_for_top_neighbor + dev_id, 2 * nx * ny * sizeof(real)));
        CUDA_RT_CALL(
            cudaMalloc(halo_buffer_for_bottom_neighbor + dev_id, 2 * nx * ny * sizeof(real)));

        CUDA_RT_CALL(
            cudaMemset(halo_buffer_for_top_neighbor[dev_id], 0, 2 * nx * ny * sizeof(real)));
        CUDA_RT_CALL(
            cudaMemset(halo_buffer_for_bottom_neighbor[dev_id], 0, 2 * nx * ny * sizeof(real)));

        CUDA_RT_CALL(
            cudaMalloc(is_top_done_computing_flags + dev_id, total_num_flags * sizeof(int)));
        CUDA_RT_CALL(
            cudaMalloc(is_bottom_done_computing_flags + dev_id, total_num_flags * sizeof(int)));

        CUDA_RT_CALL(
            cudaMemset(is_top_done_computing_flags[dev_id], 0, total_num_flags * sizeof(int)));
        CUDA_RT_CALL(
            cudaMemset(is_bottom_done_computing_flags[dev_id], 0, total_num_flags * sizeof(int)));

        // Calculate local domain boundaries
        int iz_start_global;  // My start index in the global array
        if (dev_id < num_ranks_low) {
            iz_start_global = dev_id * chunk_size_low + 1;
        } else {
            iz_start_global =
                num_ranks_low * chunk_size_low + (dev_id - num_ranks_low) * chunk_size_high + 1;
        }
        int iz_end_global = iz_start_global + chunk_size - 1;  // My last index in the global array

        int iz_start = 1;
        iz_end[dev_id] = (iz_end_global - iz_start_global + 1) + iz_start;

        // Set diriclet boundary conditions on left and right border
        //        initialize_boundaries<<<(nz / num_devices) / 128 + 1, 128>>>(
        //            a[dev_id], a_new[dev_id], PI, iz_start_global - 1, nx, ny, chunk_size + 2,
        //            nz);
        //        CUDA_RT_CALL(cudaGetLastError());

        CUDA_RT_CALL(cudaDeviceSynchronize());

        dim3 dim_grid(numSms - 2, 1, 1);
        dim3 dim_block(dim_block_x, dim_block_y, dim_block_z);

        void *kernelArgsBoundary[] = {(void *)&a_new[dev_id],
                                      (void *)&a[dev_id],
                                      (void *)&iz_start,
                                      (void *)&iz_end[dev_id],
                                      (void *)&ny,
                                      (void *)&nx,
                                      (void *)&comm_tile_size_x,
                                      (void *)&comm_tile_size_y,
                                      (void *)&num_comm_tiles_x,
                                      (void *)&num_comm_tiles_y,
                                      (void *)&iter_max,
                                      (void *)&halo_buffer_for_top_neighbor[dev_id],
                                      (void *)&halo_buffer_for_bottom_neighbor[dev_id],
                                      (void *)&halo_buffer_for_bottom_neighbor[top],
                                      (void *)&halo_buffer_for_top_neighbor[bottom],
                                      (void *)&is_top_done_computing_flags[dev_id],
                                      (void *)&is_bottom_done_computing_flags[dev_id],
                                      (void *)&is_bottom_done_computing_flags[top],
                                      (void *)&is_top_done_computing_flags[bottom],
                                      (void *)&iteration_done_flags[0]};

#pragma omp barrier
        cudaStream_t inner_domain_stream;
        cudaStream_t boundary_sync_stream;

        CUDA_RT_CALL(cudaStreamCreate(&inner_domain_stream));
        CUDA_RT_CALL(cudaStreamCreate(&boundary_sync_stream));

        const int LOCAL_ITEM_PER_THREAD = isDoubleTile ? ITEM_PER_THREAD * 2 : ITEM_PER_THREAD;
        bdimx = bdimx == 128 ? 128 : 256;
        if (isDoubleTile) {
            if (bdimx == 256) blkpsm = 1;
            if (bdimx == 128) blkpsm = min(blkpsm, 2);
        }
        int TILE_Y = LOCAL_ITEM_PER_THREAD * bdimx / TILE_X;
        if (blkpsm <= 0) blkpsm = 100;
        // int iteration=4;
        /* Host allocation Begin */
        int sm_count;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

        auto execute_kernel =
            isDoubleTile
                ?

                (bdimx == 128
                     ? (blkpsm >= 4
                            ? (useSM ? kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD,
                                                                TILE_X, 256, true, 128, curshape>
                                     : kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD,
                                                                TILE_X, 256, false, 128, curshape>)
                            : (useSM ? kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD,
                                                                TILE_X, 256, true, 128, curshape>
                                     : kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD,
                                                                TILE_X, 256, false, 128, curshape>))
                     : (blkpsm >= 2
                            ? (useSM ? kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD,
                                                                TILE_X, 256, true, 256, curshape>
                                     : kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD,
                                                                TILE_X, 256, false, 256, curshape>)
                            : (useSM
                                   ? kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD,
                                                              TILE_X, 256, true, 256, curshape>
                                   : kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD,
                                                              TILE_X, 256, false, 256, curshape>)))
                : (bdimx == 128
                       ? (blkpsm >= 4
                              ? (useSM
                                     ? kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
                                                                128, true, 128, curshape>
                                     : kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
                                                                128, false, 128, curshape>)
                              : (useSM
                                     ? kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
                                                                256, true, 128, curshape>
                                     : kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
                                                                256, false, 128, curshape>))
                       : (blkpsm >= 2
                              ? (useSM
                                     ? kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
                                                                128, true, 256, curshape>
                                     : kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
                                                                128, false, 256, curshape>)
                              : (useSM
                                     ? kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
                                                                256, true, 256, curshape>
                                     : kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
                                                                256, false, 256, curshape>)));
        int reg_folder_z = 0;
        // if(isDoubleTile)
        bool ifspill = false;
        {
            if (bdimx == 128) {
                if (blkpsm >= 4) {
                    if (ptx == 800) {
                        reg_folder_z = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128,
                                                         800, true, REAL>::val
                                             : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128,
                                                         800, false, REAL>::val;
                        ifspill = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128, 800,
                                                    true, REAL>::spill
                                        : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128, 800,
                                                    false, REAL>::spill;
                    }
                    if (ptx == 700) {
                        reg_folder_z = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128,
                                                         700, true, REAL>::val
                                             : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128,
                                                         700, false, REAL>::val;
                        ifspill = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128, 700,
                                                    true, REAL>::spill
                                        : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128, 700,
                                                    false, REAL>::spill;
                    }
                } else {
                    if (isDoubleTile) {
                        if (ptx == 800) {
                            reg_folder_z = useSM
                                               ? regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD,
                                                           256, 800, true, REAL>::val
                                               : regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD,
                                                           256, 800, false, REAL>::val;
                            ifspill = useSM ? regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD,
                                                        256, 800, true, REAL>::spill
                                            : regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD,
                                                        256, 800, false, REAL>::spill;
                        }
                        if (ptx == 700) {
                            reg_folder_z = useSM
                                               ? regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD,
                                                           256, 700, true, REAL>::val
                                               : regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD,
                                                           256, 700, false, REAL>::val;
                            ifspill = useSM ? regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD,
                                                        256, 700, true, REAL>::spill
                                            : regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD,
                                                        256, 700, false, REAL>::spill;
                        }
                    } else {
                        if (ptx == 800) {
                            reg_folder_z = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD,
                                                             256, 800, true, REAL>::val
                                                 : regfolder<HALO, curshape, 128, ITEM_PER_THREAD,
                                                             256, 800, false, REAL>::val;
                            ifspill = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 256,
                                                        800, true, REAL>::spill
                                            : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 256,
                                                        800, false, REAL>::spill;
                        }
                        if (ptx == 700) {
                            reg_folder_z = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD,
                                                             256, 700, true, REAL>::val
                                                 : regfolder<HALO, curshape, 128, ITEM_PER_THREAD,
                                                             256, 700, false, REAL>::val;
                            ifspill = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 256,
                                                        700, true, REAL>::spill
                                            : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 256,
                                                        700, false, REAL>::spill;
                        }
                    }
                }
            } else {
                if (blkpsm >= 2) {
                    if (ptx == 800) {
                        reg_folder_z = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128,
                                                         800, true, REAL>::val
                                             : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128,
                                                         800, false, REAL>::val;
                        ifspill = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128, 800,
                                                    true, REAL>::spill
                                        : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128, 800,
                                                    false, REAL>::spill;
                    }
                    if (ptx == 700) {
                        reg_folder_z = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128,
                                                         700, true, REAL>::val
                                             : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128,
                                                         700, false, REAL>::val;
                        ifspill = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128, 700,
                                                    true, REAL>::spill
                                        : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128, 700,
                                                    false, REAL>::spill;
                    }
                } else {
                    if (isDoubleTile) {
                        if (ptx == 800) {
                            reg_folder_z = useSM
                                               ? regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD,
                                                           256, 800, true, REAL>::val
                                               : regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD,
                                                           256, 800, false, REAL>::val;
                            ifspill = useSM ? regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD,
                                                        256, 800, true, REAL>::spill
                                            : regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD,
                                                        256, 800, false, REAL>::spill;
                        }
                        if (ptx == 700) {
                            reg_folder_z = useSM
                                               ? regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD,
                                                           256, 700, true, REAL>::val
                                               : regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD,
                                                           256, 700, false, REAL>::val;
                            ifspill = useSM ? regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD,
                                                        256, 700, true, REAL>::spill
                                            : regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD,
                                                        256, 700, false, REAL>::spill;
                        }
                    } else {
                        if (ptx == 800) {
                            reg_folder_z = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD,
                                                             256, 800, true, REAL>::val
                                                 : regfolder<HALO, curshape, 256, ITEM_PER_THREAD,
                                                             256, 800, false, REAL>::val;
                            ifspill = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 256,
                                                        800, true, REAL>::spill
                                            : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 256,
                                                        800, false, REAL>::spill;
                        }
                        if (ptx == 700) {
                            reg_folder_z = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD,
                                                             256, 700, true, REAL>::val
                                                 : regfolder<HALO, curshape, 256, ITEM_PER_THREAD,
                                                             256, 700, false, REAL>::val;
                            ifspill = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 256,
                                                        700, true, REAL>::spill
                                            : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 256,
                                                        700, false, REAL>::spill;
                        }
                    }
                }
            }
        }

        if (ifspill) printf("JESS\n");

        // shared memory related
        size_t executeSM = 0;

        int basic_sm_space =
            ((TILE_Y + 2 * HALO) * (TILE_X + HALO + isBOX) * (1 + HALO * 2) + 1) * sizeof(REAL);
        executeSM = basic_sm_space;

        // shared memory related
        int maxSharedMemory;
        CUDA_RT_CALL(cudaDeviceGetAttribute(&maxSharedMemory,
                                            cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
        int SharedMemoryUsed = maxSharedMemory - 2048;
        CUDA_RT_CALL(cudaFuncSetAttribute(
            execute_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemoryUsed));

        int max_sm_flder = 0;

        // printf("asdfjalskdfjaskldjfals;");

        int numBlocksPerSm_current = 100;

        executeSM += reg_folder_z * 2 * HALO * (TILE_Y + TILE_X + 2 * isBOX);

        CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocksPerSm_current, execute_kernel, bdimx, executeSM));

        if (blkpsm <= 0) blkpsm = numBlocksPerSm_current;
        numBlocksPerSm_current = min(blkpsm, numBlocksPerSm_current);

        dim3 block_dim_3(bdimx, 1, 1);
        dim3 grid_dim_3(
            nx / TILE_X, ny / TILE_Y,
            MIN(nz, MAX(1, sm_count * numBlocksPerSm_current / (nx * ny / TILE_X / TILE_Y))));
        dim3 executeBlockDim = block_dim_3;
        dim3 executeGridDim = grid_dim_3;

        if (numBlocksPerSm_current == 0) printf("JESSE 3\n");

        int perSMUsable = SharedMemoryUsed / numBlocksPerSm_current;
        int perSMValsRemaind = (perSMUsable - basic_sm_space) / sizeof(REAL);
        int reg_boundary = reg_folder_z * 2 * HALO * (TILE_Y + TILE_X + 2 * isBOX);
        max_sm_flder = (perSMValsRemaind - reg_boundary) /
                       (2 * HALO * (TILE_Y + TILE_X + 2 * isBOX) + TILE_X * TILE_Y);

        if (!useSM) max_sm_flder = 0;
        if (useSM && max_sm_flder == 0) printf("JESSE 2\n");

        int sharememory1 = 2 * HALO * (TILE_Y + TILE_X + 2 * isBOX) *
                           (max_sm_flder + reg_folder_z) * sizeof(REAL);  // boundary
        int sharememory2 = sharememory1 + sizeof(REAL) * (max_sm_flder) * (TILE_Y)*TILE_X;
        executeSM = sharememory2 + basic_sm_space;

        //        int minHeight = (max_sm_flder + reg_folder_z + 2 * NOCACHE_Z) * executeGridDim.z;

        if (executeGridDim.z * (2 * HALO + 1) > unsigned(nz)) printf("JESSE 4\n");

        REAL *input;
        CUDA_RT_CALL(cudaMalloc(&input, sizeof(REAL) * (nz * nx * ny)));

        //    cudaMemcpy(input, h_input, sizeof(REAL) * (height * width_x * width_y),
        //    cudaMemcpyHostToDevice);
        //        REAL *__var_1__;
        //        CUDA_RT_CALL(cudaMalloc(&__var_1__, sizeof(REAL) * (nz * nx * ny)));
        //        REAL *__var_2__;
        //        CUDA_RT_CALL(cudaMalloc(&__var_2__, sizeof(REAL) * (nz * nx * ny)));
        //
        size_t L2_utage = ny * nz * sizeof(REAL) * HALO * (nx / TILE_X) * 2 +
                          nx * nz * sizeof(REAL) * HALO * (ny / TILE_Y) * 2;

        REAL *l2_cache1;
        REAL *l2_cache2;
        CUDA_RT_CALL(cudaMalloc(&l2_cache1, L2_utage));
        CUDA_RT_CALL(cudaMalloc(&l2_cache2, L2_utage));

        int l_iteration = iter_max;

        const auto a_local = a[dev_id] + ny * nx;
        const auto a_new_local = a_new[dev_id] + ny * nx;

        void *KernelArgs[] = {
            (void *)&a_local,     (void *)&a_new_local, (void *)&chunk_size,
            (void *)&ny,          (void *)&nx,          (void *)&l2_cache1,
            (void *)&l2_cache2,   (void *)&l_iteration, (void *)&iteration_done_flags[0],
            (void *)&max_sm_flder};

        double start = omp_get_wtime();

        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *)execute_kernel, executeGridDim,
                                                 executeBlockDim, KernelArgs, executeSM,
                                                 inner_domain_stream));

        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *)boundary_sync_kernel, 2, dim_block,
                                                 kernelArgsBoundary, 0, boundary_sync_stream));

        CUDA_RT_CALL(cudaDeviceSynchronize());

#pragma omp barrier
        double stop = omp_get_wtime();

        if (compare_to_single_gpu) {
            CUDA_RT_CALL(cudaMemcpy(
                a_h + iz_start_global * ny * nx, a[dev_id] + ny * nx,
                std::min((nz - iz_start_global) * ny * nx, chunk_size * nx * ny) * sizeof(real),
                cudaMemcpyDeviceToHost));
        }

#pragma omp barrier

#pragma omp master
        {
            report_results(nz, ny, nx, a_ref_h, a_h, num_devices, runtime_serial_non_persistent,
                           start, stop, compare_to_single_gpu);
        }

        //        CUDA_RT_CALL(cudaFree(a_new[dev_id]));
        //        CUDA_RT_CALL(cudaFree(a[dev_id]));

        if (compare_to_single_gpu && 0 == dev_id) {
            //            CUDA_RT_CALL(cudaFreeHost(a_h));
            //            CUDA_RT_CALL(cudaFreeHost(a_ref_h));
        }
    }

    return 0;
}
// #pragma clang diagnostic pop