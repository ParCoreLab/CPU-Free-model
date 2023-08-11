/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include <cooperative_groups.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "../../include_nvshmem/PERKS-nvshmem/multi-stream-perks-nvshmem.h"

#define HALO (1)

#include "./common/cuda_common.cuh"
#include "./common/cuda_computation.cuh"
#include "./common/jacobi_cuda.cuh"
#include "./common/jacobi_reference.hpp"
#include "./common/types.hpp"
#include "./config.cuh"
#include "./perksconfig.cuh"
#include "config.cuh"

namespace cg = cooperative_groups;

namespace MultiStreamPERKSNvshmem {
__global__ void __launch_bounds__(1024, 1)
    boundary_sync_kernel(real *a_new, real *a, const int iz_start, const int iz_end, const int ny,
                         const int nx, const int iter_max, real *halo_buffer_top,
                         real *halo_buffer_bottom, uint64_t *is_done_computing_flags, const int top,
                         const int bottom, const int top_iz, const int bottom_iz,
                         volatile int *iteration_done) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    int iter = 0;
    int cur_iter_mod = 0;
    int next_iter_mod = 1;

    const int end_iz = (iz_end - 1) * ny * nx;
    //        const int end_iy = (ny) * nx;
    const int end_iy = (ny)*nx;
    const int end_ix = (nx);

    const int comm_size_iy = blockDim.y * nx;
    const int comm_size_ix = blockDim.x;

    const int comm_start_iy = (threadIdx.y) * nx;
    const int comm_start_ix = threadIdx.x;
    const int comm_start_iz = iz_start * ny * nx;

    while (iter < iter_max) {
        while (iteration_done[1] != iter) {
        }

        if (blockIdx.x == gridDim.x - 1) {
            if (cta.thread_rank() == cta.num_threads() - 1) {
                nvshmem_signal_wait_until(is_done_computing_flags + cur_iter_mod * 2,
                                          NVSHMEM_CMP_EQ, iter);
            }
            cg::sync(cta);

            for (int iy = comm_start_iy; iy < end_iy; iy += comm_size_iy) {
                int north_idx_y = comm_start_iz + (iy + (iy < (ny - 1) * nx));
                int south_idx_y = comm_start_iz + (iy - (iy > 0) * nx);

                for (int ix = comm_start_ix; ix < end_ix; ix += comm_size_ix) {
                    // this is the real index
                    real east = a[comm_start_iz + iy + ix + (ix < (nx - 1))];
                    real west = a[comm_start_iz + iy + ix - (ix > 0)];

                    real north = a[north_idx_y + ix];
                    real south = a[south_idx_y + ix];

                    real top = halo_buffer_top[cur_iter_mod * ny * nx + iy + ix];
                    real bottom = a[comm_start_iz + ny * nx + iy + ix];

                    const real first_row_val =
                        (1.0f / 6.0f) * (north + south + west + east + top + bottom);

                    a_new[comm_start_iz + iy + ix] = first_row_val;
                }
            }

            nvshmemx_putmem_signal_nbi_block(halo_buffer_bottom + (next_iter_mod)*ny * nx,
                                             a_new + ny * nx, ny * nx * sizeof(real),
                                             is_done_computing_flags + next_iter_mod * 2 + 1,
                                             iter + 1, NVSHMEM_SIGNAL_SET, top);

        } else if (blockIdx.x == gridDim.x - 2) {
            if (cta.thread_rank() == cta.num_threads() - 1) {
                nvshmem_signal_wait_until(is_done_computing_flags + cur_iter_mod * 2 + 1,
                                          NVSHMEM_CMP_EQ, iter);
            }
            cg::sync(cta);

            for (int iy = comm_start_iy; iy < end_iy; iy += comm_size_iy) {
                int north_idx_y = end_iz + (iy + (iy < (ny - 1) * nx));
                int south_idx_y = end_iz + (iy - (iy > 0) * nx);

                for (int ix = comm_start_ix; ix < end_ix; ix += comm_size_ix) {
                    real east = a[end_iz + iy + ix + (ix < (nx - 1))];
                    real west = a[end_iz + iy + ix - (ix > 0)];

                    real north = a[north_idx_y + ix];
                    real south = a[south_idx_y + ix];

                    real top = a[end_iz - ny * nx + iy + ix];
                    real bottom = halo_buffer_bottom[cur_iter_mod * ny * nx + iy + ix];

                    const real last_row_val =
                        (1.0f / 6.0f) * (north + south + west + east + top + bottom);

                    a_new[end_iz + iy + ix] = last_row_val;
                }
            }

            nvshmemx_putmem_signal_nbi_block(halo_buffer_top + next_iter_mod * ny * nx,
                                             a_new + (iz_end - 1) * nx * ny, ny * nx * sizeof(real),
                                             is_done_computing_flags + next_iter_mod * 2, iter + 1,
                                             NVSHMEM_SIGNAL_SET, bottom);
        }

        real *temp_pointer_first = a_new;
        a_new = a;
        a = temp_pointer_first;

        iter++;

        next_iter_mod = cur_iter_mod;
        cur_iter_mod = 1 - cur_iter_mod;

        cg::sync(grid);

        if (grid.thread_rank() == 0) {
            iteration_done[0] = iter;
        }

        // Might not be necessary
        //            cg::sync(grid);
    }
}
}  // namespace MultiStreamPERKSNvshmem

int MultiStreamPERKSNvshmem::init(int argc, char *argv[]) {
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

    // PERKS config
#define ITEM_PER_THREAD (8)
#define REG_FOLDER_Z (0)
    // #define TILE_X 256

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

    const int LOCAL_ITEM_PER_THREAD = isDoubleTile ? ITEM_PER_THREAD * 2 : ITEM_PER_THREAD;

    int TILE_Y = LOCAL_ITEM_PER_THREAD * bdimx / TILE_X;

#undef __PRINT__
#define PERSISTENTLAUNCH

#define REAL real

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

    if (compare_to_single_gpu) {
        CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * nz * sizeof(real)));
        CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * nz * sizeof(real)));

        runtime_serial_non_persistent =
            single_gpu(nz, ny, nx, iter_max, a_ref_h, 0, true, jacobi_kernel_single_gpu_mirror);
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

    cudaDeviceProp deviceProp{};
    CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, mype));
    //    int numSms = deviceProp.multiProcessorCount;

    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 8;
    constexpr int dim_block_z = 4;

    //    constexpr int grid_dim_x = 2;
    //    constexpr int grid_dim_y = 4;
    //    const int grid_dim_z = (numSms - 2) / (grid_dim_x * grid_dim_y);

    int total_num_flags = 4;

    const int top = mype > 0 ? mype - 1 : (npes - 1);
    const int bottom = (mype + 1) % npes;

    int iz_end_top = (top < num_ranks_low) ? chunk_size_low + 1 : chunk_size_high + 1;
    int iz_start_bottom = 0;

    if (top != mype) {
        int canAccessPeer = 0;
        CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, mype, top));
        if (canAccessPeer) {
            //           CUDA_RT_CALL(cudaDeviceEnablePeerAccess(top, 0));
        } else {
            std::cerr << "P2P access required from " << mype << " to " << top << std::endl;
        }
        if (top != bottom) {
            canAccessPeer = 0;
            CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, mype, bottom));
            if (canAccessPeer) {
                //               CUDA_RT_CALL(cudaDeviceEnablePeerAccess(bottom, 0));
            } else {
                std::cerr << "P2P access required from " << mype << " to " << bottom << std::endl;
            }
        }
    }

    nvshmem_barrier_all();

    // Using chunk_size_high so that it is same across all PEs
    a = (real *)nvshmem_malloc(nx * ny * (chunk_size_high + 2) * sizeof(real));
    a_new = (real *)nvshmem_malloc(nx * ny * (chunk_size_high + 2) * sizeof(real));

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

    nvshmem_barrier_all();

    // Initialize boundary buffers
    CUDA_RT_CALL(cudaMemcpy(halo_buffer_top, a, nx * ny * sizeof(real), cudaMemcpyDeviceToDevice));
    CUDA_RT_CALL(cudaMemcpy(halo_buffer_bottom, a + (chunk_size + 1) * ny * nx,
                            nx * ny * sizeof(real), cudaMemcpyDeviceToDevice));

    nvshmem_barrier_all();

    CUDA_RT_CALL(cudaDeviceSynchronize());

    //    dim3 comp_dim_grid(grid_dim_x, grid_dim_y, grid_dim_z);
    //    dim3 comp_dim_block(dim_block_x, dim_block_y, dim_block_z);

    dim3 comm_dim_grid(2);
    dim3 comm_dim_block(dim_block_x, dim_block_y * dim_block_z);

    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    auto execute_kernel =
        isDoubleTile
            ?

            (bdimx == 128
                 ? (blkpsm >= 4
                        ? (useSM ? kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD, TILE_X,
                                                            256, true, 128, curshape>
                                 : kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD, TILE_X,
                                                            256, false, 128, curshape>)
                        : (useSM ? kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD, TILE_X,
                                                            256, true, 128, curshape>
                                 : kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD, TILE_X,
                                                            256, false, 128, curshape>))
                 : (blkpsm >= 2
                        ? (useSM ? kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD, TILE_X,
                                                            256, true, 256, curshape>
                                 : kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD, TILE_X,
                                                            256, false, 256, curshape>)
                        : (useSM ? kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD, TILE_X,
                                                            256, true, 256, curshape>
                                 : kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD, TILE_X,
                                                            256, false, 256, curshape>)))
            : (bdimx == 128
                   ? (blkpsm >= 4
                          ? (useSM ? kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
                                                              128, true, 128, curshape>
                                   : kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
                                                              128, false, 128, curshape>)
                          : (useSM ? kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
                                                              256, true, 128, curshape>
                                   : kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
                                                              256, false, 128, curshape>))
                   : (blkpsm >= 2
                          ? (useSM ? kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
                                                              128, true, 256, curshape>
                                   : kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
                                                              128, false, 256, curshape>)
                          : (useSM ? kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
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
                    reg_folder_z = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128, 800,
                                                     true, REAL>::val
                                         : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128, 800,
                                                     false, REAL>::val;
                    ifspill = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128, 800,
                                                true, REAL>::spill
                                    : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128, 800,
                                                false, REAL>::spill;
                }
                if (ptx == 700) {
                    reg_folder_z = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128, 700,
                                                     true, REAL>::val
                                         : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128, 700,
                                                     false, REAL>::val;
                    ifspill = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128, 700,
                                                true, REAL>::spill
                                    : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128, 700,
                                                false, REAL>::spill;
                }
            } else {
                if (isDoubleTile) {
                    if (ptx == 800) {
                        reg_folder_z = useSM ? regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD,
                                                         256, 800, true, REAL>::val
                                             : regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD,
                                                         256, 800, false, REAL>::val;
                        ifspill = useSM ? regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD, 256,
                                                    800, true, REAL>::spill
                                        : regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD, 256,
                                                    800, false, REAL>::spill;
                    }
                    if (ptx == 700) {
                        reg_folder_z = useSM ? regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD,
                                                         256, 700, true, REAL>::val
                                             : regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD,
                                                         256, 700, false, REAL>::val;
                        ifspill = useSM ? regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD, 256,
                                                    700, true, REAL>::spill
                                        : regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD, 256,
                                                    700, false, REAL>::spill;
                    }
                } else {
                    if (ptx == 800) {
                        reg_folder_z = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 256,
                                                         800, true, REAL>::val
                                             : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 256,
                                                         800, false, REAL>::val;
                        ifspill = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 256, 800,
                                                    true, REAL>::spill
                                        : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 256, 800,
                                                    false, REAL>::spill;
                    }
                    if (ptx == 700) {
                        reg_folder_z = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 256,
                                                         700, true, REAL>::val
                                             : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 256,
                                                         700, false, REAL>::val;
                        ifspill = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 256, 700,
                                                    true, REAL>::spill
                                        : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 256, 700,
                                                    false, REAL>::spill;
                    }
                }
            }
        } else {
            if (blkpsm >= 2) {
                if (ptx == 800) {
                    reg_folder_z = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128, 800,
                                                     true, REAL>::val
                                         : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128, 800,
                                                     false, REAL>::val;
                    ifspill = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128, 800,
                                                true, REAL>::spill
                                    : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128, 800,
                                                false, REAL>::spill;
                }
                if (ptx == 700) {
                    reg_folder_z = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128, 700,
                                                     true, REAL>::val
                                         : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128, 700,
                                                     false, REAL>::val;
                    ifspill = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128, 700,
                                                true, REAL>::spill
                                    : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128, 700,
                                                false, REAL>::spill;
                }
            } else {
                if (isDoubleTile) {
                    if (ptx == 800) {
                        reg_folder_z = useSM ? regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD,
                                                         256, 800, true, REAL>::val
                                             : regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD,
                                                         256, 800, false, REAL>::val;
                        ifspill = useSM ? regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD, 256,
                                                    800, true, REAL>::spill
                                        : regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD, 256,
                                                    800, false, REAL>::spill;
                    }
                    if (ptx == 700) {
                        reg_folder_z = useSM ? regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD,
                                                         256, 700, true, REAL>::val
                                             : regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD,
                                                         256, 700, false, REAL>::val;
                        ifspill = useSM ? regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD, 256,
                                                    700, true, REAL>::spill
                                        : regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD, 256,
                                                    700, false, REAL>::spill;
                    }
                } else {
                    if (ptx == 800) {
                        reg_folder_z = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 256,
                                                         800, true, REAL>::val
                                             : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 256,
                                                         800, false, REAL>::val;
                        ifspill = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 256, 800,
                                                    true, REAL>::spill
                                        : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 256, 800,
                                                    false, REAL>::spill;
                    }
                    if (ptx == 700) {
                        reg_folder_z = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 256,
                                                         700, true, REAL>::val
                                             : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 256,
                                                         700, false, REAL>::val;
                        ifspill = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 256, 700,
                                                    true, REAL>::spill
                                        : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 256, 700,
                                                    false, REAL>::spill;
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
    CUDA_RT_CALL(
        cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
    int SharedMemoryUsed = maxSharedMemory - 2048;
    CUDA_RT_CALL(cudaFuncSetAttribute(execute_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      SharedMemoryUsed));

    int max_sm_flder = 0;

    // printf("asdfjalskdfjaskldjfals;");

    int numBlocksPerSm_current = 100;

    executeSM += reg_folder_z * 2 * HALO * (TILE_Y + TILE_X + 2 * isBOX);

    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm_current,
                                                               execute_kernel, bdimx, executeSM));

    if (blkpsm <= 0) blkpsm = numBlocksPerSm_current;
    numBlocksPerSm_current = min(blkpsm, numBlocksPerSm_current);

    dim3 block_dim_3(bdimx, 1, 1);
    dim3 grid_dim_3(
        nx / TILE_X, ny / TILE_Y,
        MIN(nz, MAX(1, sm_count * numBlocksPerSm_current / (nx * ny / TILE_X / TILE_Y))));
    dim3 executeBlockDim = block_dim_3;
    dim3 executeGridDim = grid_dim_3;

    if (numBlocksPerSm_current == 0) printf("JESSE 3\n");

    //    int minHeight = 0;

    int perSMUsable = SharedMemoryUsed / numBlocksPerSm_current;
    int perSMValsRemaind = (perSMUsable - basic_sm_space) / sizeof(REAL);
    int reg_boundary = reg_folder_z * 2 * HALO * (TILE_Y + TILE_X + 2 * isBOX);
    max_sm_flder = (perSMValsRemaind - reg_boundary) /
                   (2 * HALO * (TILE_Y + TILE_X + 2 * isBOX) + TILE_X * TILE_Y);

    if (!useSM) max_sm_flder = 0;
    if (useSM && max_sm_flder == 0) printf("JESSE 2\n");

    int sharememory1 = 2 * HALO * (TILE_Y + TILE_X + 2 * isBOX) * (max_sm_flder + reg_folder_z) *
                       sizeof(REAL);  // boundary
    int sharememory2 = sharememory1 + sizeof(REAL) * (max_sm_flder) * (TILE_Y)*TILE_X;
    executeSM = sharememory2 + basic_sm_space;

    //    minHeight = (max_sm_flder + reg_folder_z + 2 * NOCACHE_Z) * executeGridDim.z;

    if (executeGridDim.z * (2 * HALO + 1) > unsigned(nz)) printf("JESSE 4\n");

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

    // PERKS has no iz_start, so we include the halos
    const auto chunk_size_local = chunk_size - 2;

    void *KernelArgs[] = {(void *)&a,           (void *)&a_new,       (void *)&chunk_size_local,
                          (void *)&ny,          (void *)&nx,          (void *)&l2_cache1,
                          (void *)&l2_cache2,   (void *)&l_iteration, (void *)&iteration_done_flags,
                          (void *)&max_sm_flder};

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
                                  (void *)&iz_end_top,
                                  (void *)&iz_start_bottom,
                                  (void *)&iteration_done_flags};

    nvshmem_barrier_all();

    cudaStream_t inner_domain_stream;
    cudaStream_t boundary_sync_stream;

    CUDA_RT_CALL(cudaStreamCreate(&inner_domain_stream));
    CUDA_RT_CALL(cudaStreamCreate(&boundary_sync_stream));

    double start = MPI_Wtime();

    CUDA_RT_CALL((cudaError_t)nvshmemx_collective_launch((void *)execute_kernel, executeGridDim,
                                                         executeBlockDim, KernelArgs, executeSM,
                                                         inner_domain_stream));

    CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *)MultiStreamPERKSNvshmem::boundary_sync_kernel,
                                             comm_dim_grid, comm_dim_block, kernelArgsBoundary, 0,
                                             boundary_sync_stream));

    CUDA_RT_CALL(cudaDeviceSynchronize());
    CUDA_RT_CALL(cudaGetLastError());

    // Need to swap pointers on CPU if iteration count is odd
    if (iter_max % 2 != 1) {
        std::swap(a_new, a);
    }

    nvshmem_barrier_all();
    double stop = MPI_Wtime();
    nvshmem_barrier_all();
    bool result_correct = 1;
    if (compare_to_single_gpu) {
        CUDA_RT_CALL(cudaMemcpy(a_h + iz_start_global * ny * nx, a_new + ny * nx,
                                std::min(nz - iz_start_global, chunk_size) * nx * ny * sizeof(real),
                                cudaMemcpyDeviceToHost));

        double err = 0;
        for (int iz = iz_start_global; result_correct && (iz <= iz_end_global); ++iz) {
            for (int iy = 1; result_correct && (iy < (ny - 1)); ++iy) {
                for (int ix = 1; result_correct && (ix < (nx - 1)); ++ix) {
                    if (std::fabs(a_h[iz * ny * nx + iy * nx + ix] -
                                  a_ref_h[iz * ny * nx + iy * nx + ix]) > tol &&
                        !isnan(a_h[iz * ny * nx + iy * nx + ix])) {
                        err += std::fabs(a_h[iz * ny * nx + iy * nx + ix] -
                                         a_ref_h[iz * ny * nx + iy * nx + ix]);

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

    nvshmem_free((void *)a);
    nvshmem_free((void *)a_new);
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
