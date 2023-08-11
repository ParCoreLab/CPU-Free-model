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
// https://github.com/NVIDIA/multi-gpu-programming-models/blob/master/multi_threaded_copy_overlap/jacobi.cu

#include "../../include/no-compute/multi-threaded-copy-overlap-no-compute.cuh"

namespace BaselineMultiThreadedCopyOverlapNoCompute {
__global__ void jacobi_kernel(real *__restrict__ const a_new, const real *__restrict__ const a,
                              const int iz_start, const int iz_end, const int ny, const int nx) {
    /*
    int iz = blockIdx.z * blockDim.z + threadIdx.z + iz_start;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;

    // real local_l2_norm = 0.0;

    if (iz < iz_end && iy < (ny - 1) && ix < (nx - 1))
    {
        const real new_val = (a[iz * ny * nx + iy* nx + ix + 1] +
                              a[iz * ny * nx + iy * nx+ ix - 1] +
                              a[iz * ny * nx + (iy + 1) * nx + ix] +
                              a[iz * ny * nx + (iy - 1) * nx + ix] +
                              a[(iz + 1) * ny * nx + iy * nx+ ix] +
                              a[(iz - 1) * ny * nx + iy * nx+ ix]) /
                             real(6.0);

        a_new[iz * ny * nx + iy + ix] = new_val;

        // if (calculate_norm) {
        //     real residue = new_val - a[iy * nx + ix];
        //     local_l2_norm += residue * residue;
        // }
    }
    */

    // if (calculate_norm) {
    //     atomicAdd(l2_norm, local_l2_norm);
    // }
}

}  // namespace BaselineMultiThreadedCopyOverlapNoCompute

int BaselineMultiThreadedCopyOverlapNoCompute::init(int argc, char *argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 512);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 512);
    const int nz = get_argval<int>(argv, argv + argc, "-nz", 512);
    const bool compare_to_single_gpu = get_arg(argv, argv + argc, "-compare");

    real *a_new[MAX_NUM_DEVICES];

    real *a_ref_h;
    real *a_h;
    double runtime_serial_non_persistent = 0.0;

    int iz_end[MAX_NUM_DEVICES];

    cudaEvent_t push_top_done[2][MAX_NUM_DEVICES];
    cudaEvent_t push_bottom_done[2][MAX_NUM_DEVICES];

    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

#pragma omp parallel num_threads(num_devices)
    {
        real *a;

        cudaStream_t compute_stream;
        cudaStream_t push_top_stream;
        cudaStream_t push_bottom_stream;
        // cudaEvent_t reset_l2norm_done;

        int dev_id = omp_get_thread_num();

        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaFree(0));

        if (compare_to_single_gpu && 0 == dev_id) {
            CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * nz * sizeof(real)));
            CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * nz * sizeof(real)));

            runtime_serial_non_persistent = single_gpu(nz, ny, nx, iter_max, a_ref_h, 0, true);
        }
#pragma omp barrier
        // nz - 2 planes are distributed amongst `size` ranks in such a way
        // that each rank gets either (nz - 2) / size or (nz - 2) / size + 1 planes.
        // This optimizes load balancing when (nz - 2) % size != 0
        int chunk_size;
        int chunk_size_low = (nz - 2) / num_devices;
        int chunk_size_high = chunk_size_low + 1;
        // To calculate the number of ranks that need to compute an extra plane,
        // the following formula is derived from this equation:
        // num_ranks_low * chunk_size_low + (size - num_ranks_low) * (chunk_size_low + 1) = nz - 2
        int num_ranks_low = num_devices * chunk_size_low + num_devices -
                            (nz - 2);  // Number of ranks with chunk_size = chunk_size_low
        if (dev_id < num_ranks_low)
            chunk_size = chunk_size_low;
        else
            chunk_size = chunk_size_high;

        CUDA_RT_CALL(cudaMalloc(&a, nx * ny * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(a_new + dev_id, nx * ny * (chunk_size + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMemset(a, 0, nx * ny * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(a_new[dev_id], 0, nx * ny * (chunk_size + 2) * sizeof(real)));

        // Calculate local domain boundaries
        int iz_start_global;  // My start index in the global array
        if (dev_id < num_ranks_low) {
            iz_start_global = dev_id * chunk_size_low + 1;
        } else {
            iz_start_global =
                num_ranks_low * chunk_size_low + (dev_id - num_ranks_low) * chunk_size_high + 1;
        }

        int iz_start = 1;
        iz_end[dev_id] = iz_start + chunk_size;

        // Set diriclet boundary conditions on left and right boarder
        initialize_boundaries<<<(nz / num_devices) / 128 + 1, 128>>>(
            a, a_new[dev_id], PI, iz_start_global - 1, nx, ny, (chunk_size + 2), nz);
        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());

        int leastPriority = 0;
        int greatestPriority = leastPriority;
        CUDA_RT_CALL(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

        CUDA_RT_CALL(
            cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, leastPriority));
        CUDA_RT_CALL(
            cudaStreamCreateWithPriority(&push_top_stream, cudaStreamDefault, greatestPriority));
        CUDA_RT_CALL(
            cudaStreamCreateWithPriority(&push_bottom_stream, cudaStreamDefault, greatestPriority));

        CUDA_RT_CALL(cudaEventCreateWithFlags(push_top_done[0] + dev_id, cudaEventDisableTiming));
        CUDA_RT_CALL(
            cudaEventCreateWithFlags(push_bottom_done[0] + dev_id, cudaEventDisableTiming));
        CUDA_RT_CALL(cudaEventCreateWithFlags(push_top_done[1] + dev_id, cudaEventDisableTiming));
        CUDA_RT_CALL(
            cudaEventCreateWithFlags(push_bottom_done[1] + dev_id, cudaEventDisableTiming));
        // CUDA_RT_CALL(cudaEventCreateWithFlags(&reset_l2norm_done, cudaEventDisableTiming));

        const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);
        int canAccessPeer = 0;
        CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, top));
        if (canAccessPeer) {
            CUDA_RT_CALL(cudaDeviceEnablePeerAccess(top, 0));
        }
        const int bottom = (dev_id + 1) % num_devices;
        if (top != bottom) {
            canAccessPeer = 0;
            CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, bottom));
            if (canAccessPeer) {
                CUDA_RT_CALL(cudaDeviceEnablePeerAccess(bottom, 0));
            }
        }

        CUDA_RT_CALL(cudaDeviceSynchronize());

        constexpr int dim_block_x = 32;
        constexpr int dim_block_y = 8;
        constexpr int dim_block_z = 4;

        dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x, (ny + dim_block_y - 1) / dim_block_y,
                      (nz + (num_devices * dim_block_z) - 1) / (num_devices * dim_block_z));

        int iter = 0;

        CUDA_RT_CALL(cudaDeviceSynchronize());
#pragma omp barrier
        double start = omp_get_wtime();

        while (iter < iter_max) {
            // CUDA_RT_CALL(cudaEventRecord(reset_l2norm_done, compute_stream));
// need to wait for other threads due to std::swap(a_new[dev_id],a); and event
// sharing
#pragma omp barrier
            // Compute bulk
            CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_top_done[(iter % 2)][dev_id], 0));
            CUDA_RT_CALL(
                cudaStreamWaitEvent(compute_stream, push_bottom_done[(iter % 2)][dev_id], 0));
            BaselineMultiThreadedCopyOverlapNoCompute::jacobi_kernel<<<
                dim_grid, {dim_block_x, dim_block_y, dim_block_z}, 0, compute_stream>>>(
                a_new[dev_id], a, (iz_start + 1), (iz_end[dev_id] - 1), ny, nx);
            CUDA_RT_CALL(cudaGetLastError());

            // Compute boundaries
            // CUDA_RT_CALL(cudaStreamWaitEvent(push_top_stream, reset_l2norm_done, 0));
            CUDA_RT_CALL(
                cudaStreamWaitEvent(push_top_stream, push_bottom_done[(iter % 2)][top], 0));
            BaselineMultiThreadedCopyOverlapNoCompute::
                jacobi_kernel<<<nx / 128 + 1, 128, 0, push_top_stream>>>(a_new[dev_id], a, iz_start,
                                                                         (iz_start + 1), ny, nx);
            CUDA_RT_CALL(cudaGetLastError());

            // CUDA_RT_CALL(cudaStreamWaitEvent(push_bottom_stream, reset_l2norm_done, 0));
            CUDA_RT_CALL(
                cudaStreamWaitEvent(push_bottom_stream, push_top_done[(iter % 2)][bottom], 0));
            BaselineMultiThreadedCopyOverlapNoCompute::
                jacobi_kernel<<<nx / 128 + 1, 128, 0, push_bottom_stream>>>(
                    a_new[dev_id], a, (iz_end[dev_id] - 1), iz_end[dev_id], ny, nx);
            CUDA_RT_CALL(cudaGetLastError());

            // Apply periodic boundary conditions and exchange halo
            CUDA_RT_CALL(cudaMemcpyAsync(a_new[top] + (iz_end[top] * ny * nx),
                                         a_new[dev_id] + iz_start * ny * nx, ny * nx * sizeof(real),
                                         cudaMemcpyDeviceToDevice, push_top_stream));
            CUDA_RT_CALL(cudaEventRecord(push_top_done[((iter + 1) % 2)][dev_id], push_top_stream));

            CUDA_RT_CALL(cudaMemcpyAsync(
                a_new[bottom], a_new[dev_id] + (iz_end[dev_id] - 1) * ny * nx,
                nx * ny * sizeof(real), cudaMemcpyDeviceToDevice, push_bottom_stream));
            CUDA_RT_CALL(
                cudaEventRecord(push_bottom_done[((iter + 1) % 2)][dev_id], push_bottom_stream));

#pragma omp barrier
            std::swap(a_new[dev_id], a);
            iter++;
        }

        CUDA_RT_CALL(cudaDeviceSynchronize());
#pragma omp barrier
        double stop = omp_get_wtime();

        if (compare_to_single_gpu) {
            CUDA_RT_CALL(cudaMemcpy(
                a_h + iz_start_global * ny * nx, a + ny * nx,
                std::min((nz - iz_start_global) * ny * nx, chunk_size * ny * nx) * sizeof(real),
                cudaMemcpyDeviceToHost));
        }

#pragma omp barrier

#pragma omp master
        {
            report_results(nz, ny, nx, a_ref_h, a_h, num_devices, runtime_serial_non_persistent,
                           start, stop, compare_to_single_gpu);
        }

        // CUDA_RT_CALL(cudaEventDestroy(reset_l2norm_done));
        CUDA_RT_CALL(cudaEventDestroy(push_bottom_done[1][dev_id]));
        CUDA_RT_CALL(cudaEventDestroy(push_top_done[1][dev_id]));
        CUDA_RT_CALL(cudaEventDestroy(push_bottom_done[0][dev_id]));
        CUDA_RT_CALL(cudaEventDestroy(push_top_done[0][dev_id]));
        CUDA_RT_CALL(cudaStreamDestroy(push_bottom_stream));
        CUDA_RT_CALL(cudaStreamDestroy(push_top_stream));
        CUDA_RT_CALL(cudaStreamDestroy(compute_stream));

        CUDA_RT_CALL(cudaFree(a_new[dev_id]));
        CUDA_RT_CALL(cudaFree(a));

        if (compare_to_single_gpu && 0 == dev_id) {
            CUDA_RT_CALL(cudaFreeHost(a_h));
            CUDA_RT_CALL(cudaFreeHost(a_ref_h));
        }
    }

    return 0;
}
