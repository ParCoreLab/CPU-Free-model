/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "../../include/PERKS/multi-stream-perks.cuh"

#include "../../PERKS/config.cuh"

#include "../../PERKS/common/cuda_common.cuh"
#include "../../PERKS/common/cuda_computation.cuh"
#include "../../PERKS/common/jacobi_cuda.cuh"
#include "../../PERKS/config.cuh"
#include "../../PERKS/perksconfig.cuh"

namespace cg = cooperative_groups;

namespace MultiStreamPERKS {
__global__ void __launch_bounds__(1024, 1)
    boundary_sync_kernel(real *a_new, real *a, const int iy_start, const int iy_end, const int nx,
                         const int iter_max, volatile real *local_halo_buffer_for_top_neighbor,
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

    int iter = 0;

    int cur_iter_mod = 0;
    int next_iter_mod = 1;

    const int end_iy = (iy_end - 1) * nx;
    const int end_ix = (nx);

    const int comm_size_ix = blockDim.x;

    const int comm_start_ix = threadIdx.x;
    const int comm_start_iy = iy_start * nx;

    while (iter < iter_max) {
        if (!grid.thread_rank()) {
            while (iteration_done[1] != iter) {
            }
        }
        cg::sync(grid);
        if (blockIdx.x == gridDim.x - 1) {
            if (!cta.thread_rank()) {
                while (local_is_top_neighbor_done_writing_to_me[cur_iter_mod * 2] != iter) {
                }
            }
            cg::sync(cta);

            for (int ix = comm_start_ix; ix < end_ix; ix += comm_size_ix) {
                real east = a[comm_start_iy + ix + (ix < (nx - 1))];
                real west = a[comm_start_iy + ix - (ix > 0)];

                real north = remote_my_halo_buffer_on_top_neighbor[cur_iter_mod * nx + ix];
                real south = a[comm_start_iy + nx + ix];

                const real first_row_val = 0.25f * (east + west + north + south);

                a_new[comm_start_iy + ix] = first_row_val;
                local_halo_buffer_for_top_neighbor[nx * next_iter_mod + ix] = first_row_val;
            }

            cg::sync(cta);

            if (!cta.thread_rank()) {
                remote_am_done_writing_to_top_neighbor[next_iter_mod * 2 + 1] = iter + 1;
            }
        } else if (blockIdx.x == gridDim.x - 2) {
            if (!cta.thread_rank()) {
                while (local_is_bottom_neighbor_done_writing_to_me[cur_iter_mod * 2 + 1] != iter) {
                }
            }
            cg::sync(cta);

            for (int ix = comm_start_ix; ix < end_ix; ix += comm_size_ix) {
                real east = a[end_iy + ix + (ix < (nx - 1))];
                real west = a[end_iy + ix - (ix > 0)];

                real north = remote_my_halo_buffer_on_bottom_neighbor[cur_iter_mod * nx + ix];
                real south = a[end_iy - nx + ix];

                const real last_row_val = 0.25f * (east + west + north + south);

                a_new[end_iy + ix] = last_row_val;
                local_halo_buffer_for_bottom_neighbor[nx * next_iter_mod + ix] = last_row_val;
            }

            cg::sync(cta);

            if (!cta.thread_rank()) {
                remote_am_done_writing_to_bottom_neighbor[next_iter_mod * 2] = iter + 1;
            }
        }
        real *temp_pointer_first = a_new;
        a_new = a;
        a = temp_pointer_first;

        iter++;

        cur_iter_mod = next_iter_mod;
        next_iter_mod = 1 - cur_iter_mod;
        cg::sync(grid);
        if (grid.thread_rank() == 0) {
            iteration_done[0] = iter;
        }
        cg::sync(grid);
    }
}
}  // namespace MultiStreamPERKS

int MultiStreamPERKS::init(int argc, char *argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);
    const bool compare_to_single_gpu = get_arg(argv, argv + argc, "-compare");

    real *a[MAX_NUM_DEVICES];
    real *a_new[MAX_NUM_DEVICES];
    int iy_end[MAX_NUM_DEVICES];

    real *halo_buffer_for_top_neighbor[MAX_NUM_DEVICES];
    real *halo_buffer_for_bottom_neighbor[MAX_NUM_DEVICES];

    int *is_top_done_computing_flags[MAX_NUM_DEVICES];
    int *is_bottom_done_computing_flags[MAX_NUM_DEVICES];

    int *iteration_done_flags[MAX_NUM_DEVICES];

    real *a_ref_h;
    real *a_h;

    double runtime_serial_non_persistent = 0.0;

    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

    // PERKS config

    // 128 or 256
    int bdimx = 128;
    int blkpsm = 0;

    // damnit
    if (blkpsm <= 0) blkpsm = 100;

    bool useSM = true;
    bool isDoubleTile = true;

    // Change this later
    int ptx = 800;

    int REG_FOLDER_Y = 0;

    if (blkpsm * bdimx >= 2 * 256) {
        if (useSM) {
            if (ptx == 800)
                REG_FOLDER_Y = isDoubleTile
                                   ? (regfolder<HALO, true, 128, 800, true, real, 2 * RTILE_Y>::val)
                                   : (regfolder<HALO, true, 128, 800, true, real>::val);
            if (ptx == 700)
                REG_FOLDER_Y = isDoubleTile
                                   ? (regfolder<HALO, true, 128, 700, true, real, 2 * RTILE_Y>::val)
                                   : (regfolder<HALO, true, 128, 700, true, real>::val);
        } else {
            if (ptx == 800)
                REG_FOLDER_Y =
                    isDoubleTile ? (regfolder<HALO, true, 128, 800, false, real, 2 * RTILE_Y>::val)
                                 : (regfolder<HALO, true, 128, 800, false, real>::val);
            if (ptx == 700)
                REG_FOLDER_Y =
                    isDoubleTile ? (regfolder<HALO, true, 128, 700, false, real, 2 * RTILE_Y>::val)
                                 : (regfolder<HALO, true, 128, 700, false, real>::val);
        }
    } else {
        if (useSM) {
            if (ptx == 800)
                REG_FOLDER_Y = isDoubleTile
                                   ? (regfolder<HALO, true, 256, 800, true, real, 2 * RTILE_Y>::val)
                                   : (regfolder<HALO, true, 256, 800, true, real>::val);
            if (ptx == 700)
                REG_FOLDER_Y = isDoubleTile
                                   ? (regfolder<HALO, true, 256, 700, true, real, 2 * RTILE_Y>::val)
                                   : (regfolder<HALO, true, 256, 700, true, real>::val);
        } else {
            if (ptx == 800)
                REG_FOLDER_Y =
                    isDoubleTile ? (regfolder<HALO, true, 256, 800, false, real, 2 * RTILE_Y>::val)
                                 : (regfolder<HALO, true, 256, 800, false, real>::val);
            if (ptx == 700)
                REG_FOLDER_Y =
                    isDoubleTile ? (regfolder<HALO, true, 256, 700, false, real, 2 * RTILE_Y>::val)
                                 : (regfolder<HALO, true, 256, 700, false, real>::val);
        }
    }

    auto execute_kernel =
        isDoubleTile ? (blkpsm * bdimx >= 2 * 256
                            ? (useSM ? kernel_general_wrapper<real, 2 * RTILE_Y, HALO, 128, true>
                                     : kernel_general_wrapper<real, 2 * RTILE_Y, HALO, 128, false>)
                            : (useSM ? kernel_general_wrapper<real, 2 * RTILE_Y, HALO, 256, true>
                                     : kernel_general_wrapper<real, 2 * RTILE_Y, HALO, 256, false>))
                     : (blkpsm * bdimx >= 2 * 256
                            ? (useSM ? kernel_general_wrapper<real, RTILE_Y, HALO, 128, true>
                                     : kernel_general_wrapper<real, RTILE_Y, HALO, 128, false>)
                            : (useSM ? kernel_general_wrapper<real, RTILE_Y, HALO, 256, true>
                                     : kernel_general_wrapper<real, RTILE_Y, HALO, 256, false>));
#pragma omp parallel num_threads(num_devices)
    {
        int dev_id = omp_get_thread_num();

        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaFree(nullptr));

        if (compare_to_single_gpu && 0 == dev_id) {
            CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * sizeof(real)));
            CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * sizeof(real)));

            runtime_serial_non_persistent =
                single_gpu(nx, ny, iter_max, a_ref_h, 0, true, jacobi_kernel_single_gpu_mirror);
        }

#pragma omp barrier

        int chunk_size;
        int chunk_size_low = (ny - 2) / num_devices;
        int chunk_size_high = chunk_size_low + 1;

        // int height_per_gpu = ny / num_devices;

        cudaDeviceProp deviceProp{};
        CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, dev_id));
        int numSms = deviceProp.multiProcessorCount;

        constexpr int dim_block_x = 32;
        constexpr int dim_block_y = 32;

        constexpr int num_flags = 4;

        int num_ranks_low = num_devices * chunk_size_low + num_devices - (ny - 2);
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

        CUDA_RT_CALL(cudaMalloc(iteration_done_flags + dev_id, 2 * sizeof(int)));
        CUDA_RT_CALL(cudaMemset(iteration_done_flags[dev_id], 0, 2 * sizeof(int)));

        // There is no meaning for adding numSms more rows. PERKS overcomputes
        // after the last index in case of a non-perfect domain match to avoid
        // branching, so we are allocating more than necessary to avoid a segfault
        CUDA_RT_CALL(cudaMalloc(a + dev_id, nx * (chunk_size + numSms) * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(a_new + dev_id, nx * (chunk_size + numSms) * sizeof(real)));

        CUDA_RT_CALL(cudaMemset(a[dev_id], 0, nx * (chunk_size + numSms) * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(a_new[dev_id], 0, nx * (chunk_size + numSms) * sizeof(real)));

        CUDA_RT_CALL(cudaMalloc(halo_buffer_for_top_neighbor + dev_id, 2 * nx * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(halo_buffer_for_bottom_neighbor + dev_id, 2 * nx * sizeof(real)));

        CUDA_RT_CALL(cudaMemset(halo_buffer_for_top_neighbor[dev_id], 0, 2 * nx * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(halo_buffer_for_bottom_neighbor[dev_id], 0, 2 * nx * sizeof(real)));

        CUDA_RT_CALL(cudaMalloc(is_top_done_computing_flags + dev_id, num_flags * sizeof(int)));
        CUDA_RT_CALL(cudaMalloc(is_bottom_done_computing_flags + dev_id, num_flags * sizeof(int)));

        CUDA_RT_CALL(cudaMemset(is_top_done_computing_flags[dev_id], 0, num_flags * sizeof(int)));
        CUDA_RT_CALL(
            cudaMemset(is_bottom_done_computing_flags[dev_id], 0, num_flags * sizeof(int)));

        // Calculate local domain boundaries
        int iy_start_global;  // My start index in the global array
        if (dev_id < num_ranks_low) {
            iy_start_global = dev_id * chunk_size_low + 1;
        } else {
            iy_start_global =
                num_ranks_low * chunk_size_low + (dev_id - num_ranks_low) * chunk_size_high + 1;
        }
        int iy_end_global = iy_start_global + chunk_size - 1;  // My last index in the global array

        int iy_start = 1;
        iy_end[dev_id] = (iy_end_global - iy_start_global + 1) + iy_start;

        // Set diriclet boundary conditions on left and right border
        initialize_boundaries<<<(ny / num_devices) / 128 + 1, 128>>>(
            a[dev_id], a_new[dev_id], PI, iy_start_global - 1, nx, chunk_size + 2, ny);
        CUDA_RT_CALL(cudaGetLastError());

        CUDA_RT_CALL(cudaDeviceSynchronize());

        CUDA_RT_CALL(cudaMemcpy((void *)halo_buffer_for_top_neighbor[dev_id],
                                a[dev_id] + iy_end[dev_id] * nx, nx * sizeof(real),
                                cudaMemcpyDeviceToDevice));
        CUDA_RT_CALL(cudaMemcpy((void *)halo_buffer_for_bottom_neighbor[dev_id], a[dev_id],
                                nx * sizeof(real), cudaMemcpyDeviceToDevice));

        dim3 comm_dim_grid(2);
        dim3 comm_dim_block(dim_block_x * dim_block_y);

        // More PERKS config
        const int LOCAL_RTILE_Y = isDoubleTile ? RTILE_Y * 2 : RTILE_Y;

        int sm_count;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

        // Reserve 2 for the boundary kernel
        sm_count -= 4;

        real *L2_cache3;
        real *L2_cache4;
        size_t L2_utage_2 = sizeof(real) * (ny)*2 * (nx / bdimx) * HALO;

        CUDA_RT_CALL(cudaMalloc(&L2_cache3, L2_utage_2 * 2));
        L2_cache4 = L2_cache3 + (ny)*2 * (nx / bdimx) * HALO;

        // initialize shared memory
        int maxSharedMemory;
        CUDA_RT_CALL(cudaDeviceGetAttribute(&maxSharedMemory,
                                            cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));

        int SharedMemoryUsed = maxSharedMemory - 1024;
        CUDA_RT_CALL(cudaFuncSetAttribute(
            execute_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemoryUsed));

        size_t executeSM = 0;

        int basic_sm_space = (LOCAL_RTILE_Y + 2 * HALO) * (bdimx + 2 * HALO) + 1;

        size_t sharememory_basic = (basic_sm_space) * sizeof(real);
        executeSM = sharememory_basic;
        {
#define halo HALO
            executeSM += (HALO * 2 * ((REG_FOLDER_Y)*LOCAL_RTILE_Y + isBOX)) * sizeof(real);
#undef halo
        }

        int numBlocksPerSm_current = 1000;

        CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocksPerSm_current, execute_kernel, bdimx, executeSM));
        CUDA_RT_CALL(cudaDeviceSynchronize());
        // printf("");
        // int smbound=SharedMemoryUsed/executeSM;
        // printf("%d,%d,%d\n",numBlocksPerSm_current,blkpsm,smbound);
        if (blkpsm != 0) {
            //            blkpsm = 1;

            numBlocksPerSm_current = min(numBlocksPerSm_current, blkpsm);
        }

        dim3 block_dim(bdimx);
        dim3 grid_dim(nx / bdimx, sm_count * numBlocksPerSm_current / (nx / bdimx));

        dim3 executeBlockDim = block_dim;
        dim3 executeGridDim = grid_dim;

#define halo HALO

        size_t max_sm_flder = 0;
        int tmp0 = SharedMemoryUsed / sizeof(real) / numBlocksPerSm_current;
        int tmp1 = 2 * HALO * isBOX;
        int tmp2 = basic_sm_space;
        int tmp3 = 2 * HALO * (REG_FOLDER_Y)*LOCAL_RTILE_Y;
        int tmp4 = 2 * HALO * (bdimx + 2 * HALO);
        tmp0 = tmp0 - tmp1 - tmp2 - tmp3 - tmp4;
        tmp0 = tmp0 > 0 ? tmp0 : 0;
        max_sm_flder = (tmp0) / (bdimx + 4 * HALO) / LOCAL_RTILE_Y;
        // printf("smflder is %d\n",max_sm_flder);
        if (!useSM) max_sm_flder = 0;
        if (useSM && max_sm_flder == 0) {
            std::cout << "Jesse" << std::endl;
        }

        size_t sm_cache_size = max_sm_flder == 0 ? 0
                                                 : (max_sm_flder * LOCAL_RTILE_Y + 2 * HALO) *
                                                       (bdimx + 2 * HALO) * sizeof(real);
        size_t y_axle_halo =
            (HALO * 2 * ((max_sm_flder + REG_FOLDER_Y) * LOCAL_RTILE_Y + isBOX)) * sizeof(real);
        executeSM = sharememory_basic + y_axle_halo;
        executeSM += sm_cache_size;
        // =====================================================================================
        const auto chunk_size_local = chunk_size;

        void *kernelArgsInner[] = {(void *)&a[dev_id],
                                   (void *)&chunk_size_local,
                                   (void *)&nx,
                                   (void *)&iy_start,
                                   (void *)&iy_end[dev_id],
                                   (void *)&a_new[dev_id],
                                   (void *)&L2_cache3,
                                   (void *)&L2_cache4,
                                   (void *)&iter_max,
                                   (void *)&max_sm_flder,
                                   (void *)&iteration_done_flags[dev_id]};

        void *kernelArgsBoundary[] = {(void *)&a_new[dev_id],
                                      (void *)&a[dev_id],
                                      (void *)&iy_start,
                                      (void *)&iy_end[dev_id],
                                      (void *)&nx,
                                      (void *)&iter_max,
                                      (void *)&halo_buffer_for_top_neighbor[dev_id],
                                      (void *)&halo_buffer_for_bottom_neighbor[dev_id],
                                      (void *)&halo_buffer_for_bottom_neighbor[top],
                                      (void *)&halo_buffer_for_top_neighbor[bottom],
                                      (void *)&is_top_done_computing_flags[dev_id],
                                      (void *)&is_bottom_done_computing_flags[dev_id],
                                      (void *)&is_bottom_done_computing_flags[top],
                                      (void *)&is_top_done_computing_flags[bottom],
                                      (void *)&iteration_done_flags[dev_id]};

#pragma omp barrier
        double start = omp_get_wtime();

        cudaStream_t inner_domain_stream;
        cudaStream_t boundary_sync_stream;

        CUDA_RT_CALL(cudaStreamCreate(&inner_domain_stream));
        CUDA_RT_CALL(cudaStreamCreate(&boundary_sync_stream));

        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *)MultiStreamPERKS::boundary_sync_kernel,
                                                 comm_dim_grid, comm_dim_block, kernelArgsBoundary,
                                                 0, boundary_sync_stream));

        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *)execute_kernel, executeGridDim,
                                                 executeBlockDim, kernelArgsInner, executeSM,
                                                 inner_domain_stream));

        CUDA_RT_CALL(cudaDeviceSynchronize());

        // Need to swap pointers on CPU if iteration count is odd
        // Technically, we don't know the iteration number (since we'll be doing l2-norm)
        // Could write iter to CPU when kernel is done
        if (iter_max % 2 == 1) {
            std::swap(a_new[dev_id], a[dev_id]);
        }

#pragma omp barrier
        double stop = omp_get_wtime();

        if (compare_to_single_gpu) {
            CUDA_RT_CALL(
                cudaMemcpy(a_h + iy_start_global * nx, a[dev_id] + nx,
                           std::min((ny - iy_start_global) * nx, chunk_size * nx) * sizeof(real),
                           cudaMemcpyDeviceToHost));
        }

#pragma omp barrier

#pragma omp master
        {
            report_results(ny, nx, a_ref_h, a_h, num_devices, runtime_serial_non_persistent, start,
                           stop, compare_to_single_gpu);
        }

        CUDA_RT_CALL(cudaFree(a_new[dev_id]));
        CUDA_RT_CALL(cudaFree(a[dev_id]));
        CUDA_RT_CALL(cudaFree(halo_buffer_for_top_neighbor[dev_id]));
        CUDA_RT_CALL(cudaFree(halo_buffer_for_bottom_neighbor[dev_id]));
        CUDA_RT_CALL(cudaFree(is_top_done_computing_flags[dev_id]));
        CUDA_RT_CALL(cudaFree(is_bottom_done_computing_flags[dev_id]));
        CUDA_RT_CALL(cudaFree(iteration_done_flags[dev_id]));

        if (compare_to_single_gpu && 0 == dev_id) {
            CUDA_RT_CALL(cudaFreeHost(a_h));
            CUDA_RT_CALL(cudaFreeHost(a_ref_h));
        }
    }

    return 0;
}
