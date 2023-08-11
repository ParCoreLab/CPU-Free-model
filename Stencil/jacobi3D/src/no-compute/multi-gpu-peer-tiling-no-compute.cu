/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "../../include/no-compute/multi-gpu-peer-tiling-no-compute.cuh"

namespace cg = cooperative_groups;

namespace MultiGPUPeerTilingNoCompute {
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
                         const int nx, const int iter_max,
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

    int iter = 0;
    int cur_iter_mod = 0;
    int next_iter_mod = 1;

    // const int end_iz = (iz_end - 1) * ny * nx;
    const int end_iy = (ny - 1) * nx;
    const int end_ix = (nx - 1);

    const int comm_size_iy = blockDim.y * nx;
    const int comm_size_ix = blockDim.x;

    const int comm_start_iy = (threadIdx.y + 1) * nx;
    const int comm_start_ix = threadIdx.x + 1;
    // const int comm_start_iz = iz_start * ny * nx;

    while (iter < iter_max) {
        while (iteration_done[1] != iter) {
        }
        if (blockIdx.x == gridDim.x - 1) {
            if (!cta.thread_rank()) {
                while (local_is_top_neighbor_done_writing_to_me[cur_iter_mod * 2] != iter) {
                }
            }
            cg::sync(cta);
            /*
            for (int iy = comm_start_iy; iy < end_iy; iy += comm_size_iy)
            {
                for (int ix = comm_start_ix; ix < end_ix; ix += comm_size_ix)
                {
                    const real first_row_val = (real(1) / real(6)) * (a[comm_start_iz + iy + ix + 1]
            + a[comm_start_iz + iy + ix - 1] + a[comm_start_iz + iy + nx + ix] + a[comm_start_iz +
            iy - nx + ix] + a[comm_start_iz + ny * nx + iy + ix] +
                                                                      remote_my_halo_buffer_on_top_neighbor[cur_iter_mod
            * ny * nx + iy + ix]); a_new[comm_start_iz + iy + ix] = first_row_val;
                    local_halo_buffer_for_top_neighbor[next_iter_mod * ny * nx + iy + ix] =
            first_row_val;
                }
            }
            cg::sync(cta);
            */
            for (int iy = comm_start_iy; iy < end_iy; iy += comm_size_iy) {
                for (int ix = comm_start_ix; ix < end_ix; ix += comm_size_ix) {
                    const real first_row_val =
                        remote_my_halo_buffer_on_top_neighbor[cur_iter_mod * ny * nx + iy + ix];
                    local_halo_buffer_for_top_neighbor[next_iter_mod * ny * nx + iy + ix] =
                        first_row_val;
                }
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
            /*
            for (int iy = comm_start_iy; iy < end_iy; iy += comm_size_iy)
            {
                for (int ix = comm_start_ix; ix < end_ix; ix += comm_size_ix)
                {

                    const real last_row_val = (real(1) / real(6)) * (a[end_iz + iy + ix + 1] +
                                                                     a[end_iz + iy + ix - 1] +
                                                                     a[end_iz + iy + nx + ix] +
                                                                     a[end_iz + iy - nx + ix] +
                                                                     remote_my_halo_buffer_on_bottom_neighbor[cur_iter_mod
            * ny * nx + iy + ix] + a[end_iz - ny * nx + iy + ix]); a_new[end_iz + iy + ix] =
            last_row_val; local_halo_buffer_for_bottom_neighbor[next_iter_mod * ny * nx + iy + ix] =
            last_row_val;
                }
            }
            cg::sync(cta);
            */
            for (int iy = comm_start_iy; iy < end_iy; iy += comm_size_iy) {
                for (int ix = comm_start_ix; ix < end_ix; ix += comm_size_ix) {
                    const real last_row_val =
                        remote_my_halo_buffer_on_bottom_neighbor[cur_iter_mod * ny * nx + iy + ix];
                    local_halo_buffer_for_bottom_neighbor[next_iter_mod * ny * nx + iy + ix] =
                        last_row_val;
                }
            }
            cg::sync(cta);

            if (!cta.thread_rank()) {
                remote_am_done_writing_to_bottom_neighbor[next_iter_mod * 2] = iter + 1;
            }
        }

        real *temp_pointer = a_new;
        a_new = a;
        a = temp_pointer;

        iter++;

        next_iter_mod = cur_iter_mod;
        cur_iter_mod = 1 - cur_iter_mod;

        cg::sync(grid);

        if (!grid.thread_rank()) {
            iteration_done[0] = iter;
        }
        cg::sync(grid);
    }
}
}  // namespace MultiGPUPeerTilingNoCompute

int MultiGPUPeerTilingNoCompute::init(int argc, char *argv[]) {
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

    int *iteration_done_flags[MAX_NUM_DEVICES];

    real *a_ref_h;
    real *a_h;

    double runtime_serial_non_persistent = 0.0;

    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

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

        // int nz_per_gpu = nz / num_devices;

        cudaDeviceProp deviceProp{};
        CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, dev_id));
        int numSms = deviceProp.multiProcessorCount;

        constexpr int dim_block_x = 32;
        constexpr int dim_block_y = 8;
        constexpr int dim_block_z = 4;

        constexpr int grid_dim_x = 2;
        constexpr int grid_dim_y = 4;
        const int grid_dim_z = (numSms - 2) / (grid_dim_x * grid_dim_y);

        constexpr int total_num_flags = 4;

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

        CUDA_RT_CALL(cudaMalloc(iteration_done_flags + dev_id, 2 * sizeof(int)));
        CUDA_RT_CALL(cudaMemset(iteration_done_flags[dev_id], 0, 2 * sizeof(int)));

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
        initialize_boundaries<<<(nz / num_devices) / 128 + 1, 128>>>(
            a[dev_id], a_new[dev_id], PI, iz_start_global - 1, nx, ny, chunk_size + 2, nz);
        CUDA_RT_CALL(cudaGetLastError());

        CUDA_RT_CALL(cudaDeviceSynchronize());

        dim3 comp_dim_grid(grid_dim_x, grid_dim_y, grid_dim_z);
        dim3 comp_dim_block(dim_block_x, dim_block_y, dim_block_z);

        dim3 comm_dim_grid(2);
        dim3 comm_dim_block(dim_block_x, dim_block_y * dim_block_z);

        void *kernelArgsInner[] = {(void *)&a_new[dev_id],
                                   (void *)&a[dev_id],
                                   (void *)&iz_start,
                                   (void *)&iz_end[dev_id],
                                   (void *)&ny,
                                   (void *)&nx,
                                   (void *)&iter_max,
                                   (void *)&iteration_done_flags[dev_id]};

        void *kernelArgsBoundary[] = {(void *)&a_new[dev_id],
                                      (void *)&a[dev_id],
                                      (void *)&iz_start,
                                      (void *)&iz_end[dev_id],
                                      (void *)&ny,
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

        // THE KERNELS ARE SERIALIZED!
        // perhaps only on V100
        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *)MultiGPUPeerTilingNoCompute::jacobi_kernel,
                                                 comp_dim_grid, comp_dim_block, kernelArgsInner, 0,
                                                 inner_domain_stream));

        CUDA_RT_CALL(cudaLaunchCooperativeKernel(
            (void *)MultiGPUPeerTilingNoCompute::boundary_sync_kernel, comm_dim_grid,
            comm_dim_block, kernelArgsBoundary, 0, boundary_sync_stream));

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
