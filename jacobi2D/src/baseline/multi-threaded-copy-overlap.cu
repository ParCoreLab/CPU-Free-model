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

#include "../../include/baseline/multi-threaded-copy-overlap.cuh"

namespace BaselineMultiThreadedCopyOverlap {
__global__ void jacobi_kernel(real* __restrict__ const a_new, const real* __restrict__ const a,
                              const int iy_start, const int iy_end, const int nx) {
    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    // real local_l2_norm = 0.0;

    if (iy < iy_end && ix < (nx - 1)) {
        const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                     a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
        a_new[iy * nx + ix] = new_val;

        // if (calculate_norm) {
        //     real residue = new_val - a[iy * nx + ix];
        //     local_l2_norm += residue * residue;
        // }
    }

    // if (calculate_norm) {
    //     atomicAdd(l2_norm, local_l2_norm);
    // }
}

}  // namespace BaselineMultiThreadedCopyOverlap

int BaselineMultiThreadedCopyOverlap::init(int argc, char* argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);
    const bool compare_to_single_gpu = get_arg(argv, argv + argc, "-compare");

    real* a_new[MAX_NUM_DEVICES];

    real* a_ref_h;
    real* a_h;
    double runtime_serial_non_persistent = 0.0;

    int iy_end[MAX_NUM_DEVICES];

    cudaEvent_t push_top_done[2][MAX_NUM_DEVICES];
    cudaEvent_t push_bottom_done[2][MAX_NUM_DEVICES];

    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

#pragma omp parallel num_threads(num_devices)
    {
        real* a;

        cudaStream_t compute_stream;
        cudaStream_t push_top_stream;
        cudaStream_t push_bottom_stream;
        // cudaEvent_t reset_l2norm_done;

        int dev_id = omp_get_thread_num();

        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaFree(0));

        if (compare_to_single_gpu && 0 == dev_id) {
            CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * sizeof(real)));
            CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * sizeof(real)));

            runtime_serial_non_persistent = single_gpu(nx, ny, iter_max, a_ref_h, 0, true);
        }
#pragma omp barrier
        // ny - 2 rows are distributed amongst `size` ranks in such a way
        // that each rank gets either (ny - 2) / size or (ny - 2) / size + 1 rows.
        // This optimizes load balancing when (ny - 2) % size != 0
        int chunk_size;
        int chunk_size_low = (ny - 2) / num_devices;
        int chunk_size_high = chunk_size_low + 1;
        // To calculate the number of ranks that need to compute an extra row,
        // the following formula is derived from this equation:
        // num_ranks_low * chunk_size_low + (size - num_ranks_low) * (chunk_size_low + 1) = ny - 2
        int num_ranks_low = num_devices * chunk_size_low + num_devices -
                            (ny - 2);  // Number of ranks with chunk_size = chunk_size_low
        if (dev_id < num_ranks_low)
            chunk_size = chunk_size_low;
        else
            chunk_size = chunk_size_high;

        CUDA_RT_CALL(cudaMalloc(&a, nx * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(a_new + dev_id, nx * (chunk_size + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMemset(a, 0, nx * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(a_new[dev_id], 0, nx * (chunk_size + 2) * sizeof(real)));

        // Calculate local domain boundaries
        int iy_start_global;  // My start index in the global array
        if (dev_id < num_ranks_low) {
            iy_start_global = dev_id * chunk_size_low + 1;
        } else {
            iy_start_global =
                num_ranks_low * chunk_size_low + (dev_id - num_ranks_low) * chunk_size_high + 1;
        }

        int iy_start = 1;
        iy_end[dev_id] = iy_start + chunk_size;

        // Set diriclet boundary conditions on left and right boarder
        initialize_boundaries<<<(ny / num_devices) / 128 + 1, 128>>>(
            a, a_new[dev_id], PI, iy_start_global - 1, nx, (chunk_size + 2), ny);
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
        constexpr int dim_block_y = 32;
        dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x,
                      (ny + (num_devices * dim_block_y) - 1) / (num_devices * dim_block_y), 1);

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
            jacobi_kernel<<<dim_grid, {dim_block_x, dim_block_y, 1}, 0, compute_stream>>>(
                a_new[dev_id], a, (iy_start + 1), (iy_end[dev_id] - 1), nx);
            CUDA_RT_CALL(cudaGetLastError());

            // Compute boundaries
            // CUDA_RT_CALL(cudaStreamWaitEvent(push_top_stream, reset_l2norm_done, 0));
            CUDA_RT_CALL(
                cudaStreamWaitEvent(push_top_stream, push_bottom_done[(iter % 2)][top], 0));
            jacobi_kernel<<<nx / 128 + 1, 128, 0, push_top_stream>>>(a_new[dev_id], a, iy_start,
                                                                     (iy_start + 1), nx);
            CUDA_RT_CALL(cudaGetLastError());

            // CUDA_RT_CALL(cudaStreamWaitEvent(push_bottom_stream, reset_l2norm_done, 0));
            CUDA_RT_CALL(
                cudaStreamWaitEvent(push_bottom_stream, push_top_done[(iter % 2)][bottom], 0));
            jacobi_kernel<<<nx / 128 + 1, 128, 0, push_bottom_stream>>>(
                a_new[dev_id], a, (iy_end[dev_id] - 1), iy_end[dev_id], nx);
            CUDA_RT_CALL(cudaGetLastError());

            // Apply periodic boundary conditions and exchange halo
            CUDA_RT_CALL(cudaMemcpyAsync(a_new[top] + (iy_end[top] * nx),
                                         a_new[dev_id] + iy_start * nx, nx * sizeof(real),
                                         cudaMemcpyDeviceToDevice, push_top_stream));
            CUDA_RT_CALL(cudaEventRecord(push_top_done[((iter + 1) % 2)][dev_id], push_top_stream));

            CUDA_RT_CALL(cudaMemcpyAsync(a_new[bottom], a_new[dev_id] + (iy_end[dev_id] - 1) * nx,
                                         nx * sizeof(real), cudaMemcpyDeviceToDevice,
                                         push_bottom_stream));
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
            CUDA_RT_CALL(
                cudaMemcpy(a_h + iy_start_global * nx, a + nx,
                           std::min((ny - iy_start_global) * nx, chunk_size * nx) * sizeof(real),
                           cudaMemcpyDeviceToHost));
        }

#pragma omp barrier

#pragma omp master
        {
            report_results(ny, nx, a_ref_h, a_h, num_devices, runtime_serial_non_persistent, start,
                           stop, compare_to_single_gpu);
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
