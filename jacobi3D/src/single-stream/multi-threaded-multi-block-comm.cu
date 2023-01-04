/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "../../include/single-stream/multi-threaded-multi-block-comm.cuh"

namespace cg = cooperative_groups;

namespace SSMultiThreadedMultiBlockComm
{

    __global__ void __launch_bounds__(1024, 1)
        jacobi_kernel(real *a_new, real *a, const int iz_start, const int iz_end, const int ny, const int nx,
                      const int comm_sm_count_per_layer, const int comm_block_count_per_sm, const int comp_block_count_per_sm,
                      const int tile_count_y, const int tile_count_x, const int iter_max,
                      volatile real *local_halo_buffer_for_top_neighbor,
                      volatile real *local_halo_buffer_for_bottom_neighbor,
                      volatile const real *remote_my_halo_buffer_on_top_neighbor,
                      volatile const real *remote_my_halo_buffer_on_bottom_neighbor,
                      volatile int *local_is_top_neighbor_done_writing_to_me,
                      volatile int *local_is_bottom_neighbor_done_writing_to_me,
                      volatile int *remote_am_done_writing_to_top_neighbor,
                      volatile int *remote_am_done_writing_to_bottom_neighbor)
    {
        cg::thread_block cta = cg::this_thread_block();
        cg::grid_group grid = cg::this_grid();

        int iter = 0;
        int cur_iter_mod = 0;
        int next_iter_mod = 1;

        const int comp_start_iz = ((blockIdx.x * comp_block_count_per_sm / (tile_count_x * tile_count_y)) * blockDim.z + threadIdx.z + iz_start + 1);
        const int comp_start_iy = ((((blockIdx.x * comp_block_count_per_sm) / tile_count_x) % tile_count_y) * blockDim.y + threadIdx.y + 1);
        const int comp_start_ix = ((blockIdx.x * comp_block_count_per_sm) % tile_count_x) * blockDim.x + threadIdx.x + 1;

        const int comm_block_id = (gridDim.x - 1) - blockIdx.x;
        const int comm_start_block_y = (((comm_block_id * comm_block_count_per_sm) / tile_count_x) * blockDim.y);
        const int comm_start_block_x = ((comm_block_id * comm_block_count_per_sm) % tile_count_x) * blockDim.x;
        const int comm_start_iy = comm_start_block_y + (threadIdx.y + 1);
        const int comm_start_ix = comm_start_block_x + threadIdx.x + 1;

        while (iter < iter_max)
        {
            if (comm_block_id < comm_sm_count_per_layer)
            {
                if (!cta.thread_rank())
                {
                    while (local_is_top_neighbor_done_writing_to_me[cur_iter_mod * 2 * comm_sm_count_per_layer + comm_block_id] !=
                           iter)
                    {
                    }
                }
                cg::sync(cta);

                int block_count = 0;
                int iy = comm_start_iy;
                int ix = comm_start_ix;

                for (; block_count < comm_block_count_per_sm && iy < (ny - 1); iy += blockDim.y)
                {
                    for (; block_count < comm_block_count_per_sm && ix < (nx - 1); ix += blockDim.x)
                    {
                        const real first_row_val = (real(1) / real(6)) * (a[iz_start * ny * nx + iy * nx + ix + 1] +
                                                                          a[iz_start * ny * nx + iy * nx + ix - 1] +
                                                                          a[iz_start * ny * nx + (iy + 1) * nx + ix] +
                                                                          a[iz_start * ny * nx + (iy - 1) * nx + ix] +
                                                                          a[(iz_start + 1) * ny * nx + iy * nx + ix] +
                                                                          remote_my_halo_buffer_on_top_neighbor[cur_iter_mod * ny * nx + iy * nx + ix]);
                        a_new[iz_start * ny * nx + iy * nx + ix] = first_row_val;
                        local_halo_buffer_for_top_neighbor[next_iter_mod * ny * nx + iy * nx + ix] = first_row_val;
                        block_count++;
                    }
                    block_count += (block_count < comp_block_count_per_sm) && !(ix < (nx - 1));
                    ix = (threadIdx.x + 1);
                }
                cg::sync(cta);

                if (!cta.thread_rank())
                {
                    remote_am_done_writing_to_top_neighbor[next_iter_mod * 2 * comm_sm_count_per_layer + comm_sm_count_per_layer + comm_block_id] = iter + 1;
                }
            }
            else if (comm_block_id < 2 * comm_sm_count_per_layer)
            {
                if (!cta.thread_rank())
                {
                    while (
                        local_is_bottom_neighbor_done_writing_to_me[cur_iter_mod * 2 * comm_sm_count_per_layer + comm_sm_count_per_layer + comm_block_id - comm_sm_count_per_layer] !=
                        iter)
                    {
                    }
                }
                cg::sync(cta);

                int block_count = 0;
                int iy = comm_start_iy;
                int ix = comm_start_ix;

                for (; block_count < comm_block_count_per_sm && iy < (ny - 1); iy += blockDim.y)
                {
                    for (; block_count < comm_block_count_per_sm && ix < (nx - 1); ix += blockDim.x)
                    {
                        const real last_row_val = (real(1) / real(6)) * (a[(iz_end - 1) * ny * nx + iy * nx + ix + 1] +
                                                                         a[(iz_end - 1) * ny * nx + iy * nx + ix - 1] +
                                                                         a[(iz_end - 1) * ny * nx + (iy + 1) * nx + ix] +
                                                                         a[(iz_end - 1) * ny * nx + (iy - 1) * nx + ix] +
                                                                         remote_my_halo_buffer_on_bottom_neighbor[cur_iter_mod * ny * nx + iy * nx + ix] +
                                                                         a[(iz_end - 2) * ny * nx + iy * nx + ix]);
                        a_new[(iz_end - 1) * ny * nx + iy * nx + ix] = last_row_val;
                        local_halo_buffer_for_bottom_neighbor[next_iter_mod * ny * nx + iy * nx + ix] = last_row_val;
                        block_count++;
                    }
                    block_count += (block_count < comp_block_count_per_sm) && !(ix < (nx - 1));
                    ix = (threadIdx.x + 1);
                }

                cg::sync(cta);

                if (!cta.thread_rank())
                {
                    remote_am_done_writing_to_bottom_neighbor[next_iter_mod * 2 * comm_sm_count_per_layer + comm_block_id - comm_sm_count_per_layer] =
                        iter + 1;
                }
            }
            else
            {
                int iz = comp_start_iz;
                int iy = comp_start_iy;
                int ix = comp_start_ix;
                int block_count = 0;

                for (; block_count < comp_block_count_per_sm && iz < (iz_end - 1); iz += blockDim.z)
                {
                    for (; block_count < comp_block_count_per_sm && iy < (ny - 1); iy += blockDim.y)
                    {
                        for (; block_count < comp_block_count_per_sm && ix < (nx - 1); ix += blockDim.x)
                        {
                            a_new[iz * ny * nx + iy * nx + ix] = (real(1) / real(6)) *
                                                                 (a[iz * ny * nx + iy * nx + ix + 1] + a[iz * ny * nx + iy * nx + ix - 1] +
                                                                  a[iz * ny * nx + (iy + 1) * nx + ix] + a[iz * ny * nx + (iy - 1) * nx + ix] +
                                                                  a[(iz + 1) * ny * nx + iy * nx + ix] + a[(iz - 1) * ny * nx + iy * nx + ix]);
                            block_count++;
                        }
                        block_count += (block_count < comp_block_count_per_sm) && !(ix < (nx - 1));
                        ix = (threadIdx.x + 1);
                    }
                    iy = (threadIdx.y + 1);
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
} // namespace SSMultiThreadedMultiBlockComm

int SSMultiThreadedMultiBlockComm::init(int argc, char *argv[])
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

    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

#pragma omp parallel num_threads(num_devices)
    {
        int dev_id = omp_get_thread_num();

        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaFree(nullptr));

        if (compare_to_single_gpu && 0 == dev_id)
        {
            CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * nz * sizeof(real)));
            CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * nz * sizeof(real)));

            runtime_serial_non_persistent = single_gpu(nz, ny, nx, iter_max, a_ref_h, 0, true);
        }

#pragma omp barrier

        int chunk_size;
        int chunk_size_low = (nz - 2) / num_devices;
        int chunk_size_high = chunk_size_low + 1;

        int num_ranks_low = num_devices * chunk_size_low + num_devices - (nz - 2);
        if (dev_id < num_ranks_low)
            chunk_size = chunk_size_low;
        else
            chunk_size = chunk_size_high;

        constexpr int num_threads_per_block = 1024;
        cudaDeviceProp deviceProp{};
        CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, dev_id));
        int numSms = deviceProp.multiProcessorCount;

        const int dim_block_x = nx >= num_threads_per_block ? num_threads_per_block : (int)pow(2, ceil(log2(nx)));
        const int dim_block_y = ny >= (num_threads_per_block / dim_block_x) ? (num_threads_per_block / dim_block_x) : (int)pow(2, ceil(log2(ny)));
        const int dim_block_z = chunk_size >= (num_threads_per_block / (dim_block_x * dim_block_y)) ? (num_threads_per_block / (dim_block_x * dim_block_y)) : (int)pow(2, ceil(log2(ny)));

        const int tile_count_x = nx / (dim_block_x) + (nx % (dim_block_x) != 0);
        const int tile_count_y = ny / (dim_block_y) + (ny % (dim_block_y) != 0);
        const int tile_count_z = chunk_size / (dim_block_z) + (chunk_size % (dim_block_z) != 0);

        const int comm_layer_tile_count = tile_count_x * tile_count_y;
        const int comp_total_tile_count = tile_count_x * tile_count_y * tile_count_z;

        const int comm_sm_count_per_layer = comm_layer_tile_count < 1 ? comm_layer_tile_count : 1;
        const int comp_sm_count = comp_total_tile_count < numSms - 2 * comm_sm_count_per_layer ? comp_total_tile_count : numSms - 2 * comm_sm_count_per_layer;

        int total_num_flags = 2 * 2 * comm_sm_count_per_layer;

        const int comp_block_count_per_sm = comp_total_tile_count / comp_sm_count + (comp_total_tile_count % comp_sm_count != 0);
        const int comm_block_count_per_sm = comm_layer_tile_count / comm_sm_count_per_layer + (comm_layer_tile_count % comm_sm_count_per_layer != 0);

        const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);
        const int bottom = (dev_id + 1) % num_devices;

        if (top != dev_id)
        {
            int canAccessPeer = 0;
            CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, top));
            if (canAccessPeer)
            {
                CUDA_RT_CALL(cudaDeviceEnablePeerAccess(top, 0));
            }
            else
            {
                std::cerr << "P2P access required from " << dev_id << " to " << top << std::endl;
            }
            if (top != bottom)
            {
                canAccessPeer = 0;
                CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, bottom));
                if (canAccessPeer)
                {
                    CUDA_RT_CALL(cudaDeviceEnablePeerAccess(bottom, 0));
                }
                else
                {
                    std::cerr << "P2P access required from " << dev_id << " to " << bottom
                              << std::endl;
                }
            }
        }

#pragma omp barrier

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
        int iz_start_global; // My start index in the global array
        if (dev_id < num_ranks_low)
        {
            iz_start_global = dev_id * chunk_size_low + 1;
        }
        else
        {
            iz_start_global =
                num_ranks_low * chunk_size_low + (dev_id - num_ranks_low) * chunk_size_high + 1;
        }
        int iz_end_global = iz_start_global + chunk_size - 1; // My last index in the global array

        int iz_start = 1;
        iz_end[dev_id] = (iz_end_global - iz_start_global + 1) + iz_start;

        initialize_boundaries<<<(nz / num_devices) / 128 + 1, 128>>>(
            a_new[dev_id], a[dev_id], PI, iz_start_global - 1, nx, ny, chunk_size + 2, nz);
        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());

        CUDA_RT_CALL(cudaMemcpy((void *)halo_buffer_for_top_neighbor[top], a[dev_id], nx * ny * sizeof(real), cudaMemcpyDeviceToDevice));
        CUDA_RT_CALL(cudaMemcpy((void *)halo_buffer_for_bottom_neighbor[bottom], a[dev_id] + iz_end[dev_id] * ny * nx, nx * ny * sizeof(real), cudaMemcpyDeviceToDevice));

        dim3 dim_grid(comp_sm_count + comm_sm_count_per_layer * 2);
        dim3 dim_block(dim_block_x, dim_block_y, dim_block_z);

        void *kernelArgs[] = {(void *)&a_new[dev_id],
                              (void *)&a[dev_id],
                              (void *)&iz_start,
                              (void *)&iz_end[dev_id],
                              (void *)&ny,
                              (void *)&nx,
                              (void *)&comm_sm_count_per_layer,
                              (void *)&comm_block_count_per_sm,
                              (void *)&comp_block_count_per_sm,
                              (void *)&tile_count_y,
                              (void *)&tile_count_x,
                              (void *)&iter_max,
                              (void *)&halo_buffer_for_top_neighbor[dev_id],
                              (void *)&halo_buffer_for_bottom_neighbor[dev_id],
                              (void *)&halo_buffer_for_bottom_neighbor[top],
                              (void *)&halo_buffer_for_top_neighbor[bottom],
                              (void *)&is_top_done_computing_flags[dev_id],
                              (void *)&is_bottom_done_computing_flags[dev_id],
                              (void *)&is_bottom_done_computing_flags[top],
                              (void *)&is_top_done_computing_flags[bottom]};

#pragma omp barrier
        double start = omp_get_wtime();

        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *)SSMultiThreadedMultiBlockComm::jacobi_kernel,
                                                 dim_grid, dim_block, kernelArgs, 0, nullptr));

        CUDA_RT_CALL(cudaDeviceSynchronize());
        CUDA_RT_CALL(cudaGetLastError());

        // Need to swap pointers on CPU if iteration count is odd
        // Technically, we don't know the iteration number (since we'll be doing
        // l2-norm) Could write iter to CPU when kernel is done
        if (iter_max % 2 == 1)
        {
            std::swap(a_new[dev_id], a[dev_id]);
        }

#pragma omp barrier
        double stop = omp_get_wtime();

        if (compare_to_single_gpu)
        {
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

        if (compare_to_single_gpu && 0 == dev_id)
        {
            CUDA_RT_CALL(cudaFreeHost(a_h));
            CUDA_RT_CALL(cudaFreeHost(a_ref_h));
        }
    }

    return 0;
}