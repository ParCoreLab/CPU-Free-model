/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

/*
 * This sample implements a conjugate gradient solver on multiple GPU using
 * Unified Memory optimized prefetching and usage hints.
 *
 */

// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <utility>

#include <omp.h>

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "../../include_nvshmem/common.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

// This should only be run with a single GPU
int SingleGPUDiscreteStandard::init(int *device_csrRowIndices, int *device_csrColIndices,
                                    real *device_csrVal, const int num_rows, const int nnz,
                                    bool matrix_is_zero_indexed, const int num_devices,
                                    const int iter_max, real *x_final_result,
                                    const double single_gpu_runtime, bool compare_to_single_gpu,
                                    bool compare_to_cpu, real *x_ref_single_gpu, real *x_ref_cpu) {
    // This version should be run with 1 GPU only but adding this check here just in case.
    int mype = nvshmem_my_pe();

    if (mype == 0) {
        bool run_as_separate_version = true;

        double single_gpu_runtime = SingleGPUDiscreteStandard::run_single_gpu(
            iter_max, device_csrRowIndices, device_csrColIndices, device_csrVal, x_ref_single_gpu,
            num_rows, nnz, matrix_is_zero_indexed, run_as_separate_version);

        printf("Execution time: %8.4f s\n", single_gpu_runtime);
    }

    return 0;
}