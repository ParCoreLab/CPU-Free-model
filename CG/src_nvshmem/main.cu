#include <iostream>

#include "../include_nvshmem/baseline/discrete-pipelined-nvshmem.cuh"
#include "../include_nvshmem/baseline/discrete-standard-nvshmem.cuh"
#include "../include_nvshmem/baseline/persistent-standard-nvshmem.cuh"
#include "../include_nvshmem/profiling/discrete-pipelined-nvshmem.cuh"
#include "../include_nvshmem/profiling/discrete-standard-nvshmem.cuh"
#include "../include_nvshmem/single-stream/pipelined-multi-overlap.cuh"
#include "../include_nvshmem/single-stream/pipelined-nvshmem.cuh"
#include "../include_nvshmem/single-stream/standard-saxpy-overlap-nvshmem.cuh"

#include "../include_nvshmem/common.h"

#include <mpi.h>

using std::make_pair;

int main(int argc, char *argv[]) {
    const std::array versions{
        make_pair("Baseline Discrete Standard NVSHMEM", BaselineDiscreteStandardNVSHMEM::init),
        make_pair("Baseline Discrete Pipelined NVSHMEM", BaselineDiscretePipelinedNVSHMEM::init),
        make_pair("Baseline Persistent Standard NVSHMEM", BaselinePersistentStandardNVSHMEM::init),
        make_pair("Single Stream Pipelined NVSHMEM", SingleStreamPipelinedNVSHMEM::init),
        make_pair("Single Stream Pipelined Multi Overlap NVSHMEM",
                  SingleStreamPipelinedMultiOverlapNVSHMEM::init),
        make_pair("Single Stream Standard Saxpy Overlap NVSHMEM",
                  SingleStreamStandardSaxpyOverlapNVSHMEM::init),
        make_pair("Profiling Discrete Standard NVSHMEM", ProfilingDiscreteStandardNVSHMEM::init),
        make_pair("Profiling Discrete Pipelined NVSHMEM", ProfilingDiscretePipelinedNVSHMEM::init),
    };

    const int selection = get_argval<int>(argv, argv + argc, "-v", 0);
    const bool silent = get_arg(argv, argv + argc, "-s");
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 10000);

    // How many times to run the selected version
    const int num_runs = get_argval<int>(argv, argv + argc, "-num_runs", 1);

    std::string matrix_path_str = get_argval<std::string>(argv, argv + argc, "-matrix_path", "");
    bool compare_to_single_gpu = get_arg(argv, argv + argc, "-compare-single-gpu");
    bool compare_to_cpu = get_arg(argv, argv + argc, "-compare-cpu");

    char *matrix_path_char = const_cast<char *>(matrix_path_str.c_str());
    bool generate_random_tridiag_matrix = matrix_path_str.empty();

    int num_devices = 0;
    double single_gpu_runtime;

    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

    int num_rows = 0;
    int num_cols = 0;
    int nnz = 0;
    bool matrix_is_zero_indexed;

    int *host_csrRowIndices = NULL;
    int *host_csrColIndices = NULL;
    real *host_csrVal = NULL;

    real *x_ref_single_gpu = NULL;
    real *x_final_result = NULL;

    real *s_cpu = NULL;
    real *r_cpu = NULL;
    real *p_cpu = NULL;
    real *x_ref_cpu = NULL;

    int *device_csrRowIndices = NULL;
    int *device_csrColIndices = NULL;
    real *device_csrVal = NULL;

    int rank = 0, size = 1;
    MPI_CALL(MPI_Init(&argc, &argv));
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));

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
        fprintf(stderr,
                "ERROR Number of visible devices (%d) is less than number of ranks on the "
                "node (%d)!\n",
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

    if (generate_random_tridiag_matrix) {
        num_rows = 10485760 * 2;
        num_cols = num_rows;

        nnz = (num_rows - 2) * 3 + 4;

        host_csrRowIndices = (int *)malloc(sizeof(int) * (num_rows + 1));
        host_csrColIndices = (int *)malloc(sizeof(int) * nnz);
        host_csrVal = (real *)malloc(sizeof(real) * nnz);

        /* Generate a random tridiagonal symmetric matrix in CSR format */
        genTridiag(host_csrRowIndices, host_csrColIndices, host_csrVal, num_rows, nnz);
    } else {
        if (loadMMSparseMatrix<real>(matrix_path_char, 'd', true, &num_rows, &num_cols, &nnz,
                                     &host_csrVal, &host_csrRowIndices, &host_csrColIndices,
                                     true)) {
            exit(EXIT_FAILURE);
        }
    }

    CUDA_RT_CALL(cudaMallocHost(&x_final_result, num_rows * sizeof(real)));

    // Check if matrix is 0 or 1 indexed
    int index_base = host_csrRowIndices[0];

    if (index_base == 1) {
        matrix_is_zero_indexed = false;
    } else if (index_base == 0) {
        matrix_is_zero_indexed = true;
    }

    CUDA_RT_CALL(cudaMalloc((void **)&device_csrRowIndices, sizeof(int) * (num_rows + 1)));
    CUDA_RT_CALL(cudaMalloc((void **)&device_csrColIndices, sizeof(int) * nnz));
    CUDA_RT_CALL(cudaMalloc((void **)&device_csrVal, sizeof(real) * nnz));

    CUDA_RT_CALL(cudaMemcpy(device_csrRowIndices, host_csrRowIndices, sizeof(int) * (num_rows + 1),
                            cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(device_csrColIndices, host_csrColIndices, sizeof(int) * nnz,
                            cudaMemcpyHostToDevice));
    CUDA_RT_CALL(
        cudaMemcpy(device_csrVal, host_csrVal, sizeof(real) * nnz, cudaMemcpyHostToDevice));

    if (compare_to_single_gpu) {
        CUDA_RT_CALL(cudaMallocHost(&x_ref_single_gpu, num_rows * sizeof(real)));

        single_gpu_runtime = SingleGPUDiscreteStandard::run_single_gpu(
            iter_max, device_csrRowIndices, device_csrColIndices, device_csrVal, x_ref_single_gpu,
            num_rows, nnz, matrix_is_zero_indexed);
    }

    if (compare_to_cpu) {
        s_cpu = (real *)malloc(sizeof(real) * num_rows);
        r_cpu = (real *)malloc(sizeof(real) * num_rows);
        p_cpu = (real *)malloc(sizeof(real) * num_rows);

        CUDA_RT_CALL(cudaMallocHost(&x_ref_cpu, num_rows * sizeof(real)));

        for (int i = 0; i < num_rows; i++) {
            r_cpu[i] = 1.0;
            s_cpu[i] = 0.0;
            x_ref_cpu[i] = 0.0;
        }

        CPU::cpuConjugateGrad(iter_max, host_csrRowIndices, host_csrColIndices, host_csrVal,
                              x_ref_cpu, s_cpu, p_cpu, r_cpu, nnz, num_rows, tol,
                              matrix_is_zero_indexed);
    }

    auto &selected = versions[selection];

    if (!silent) {
        std::cout << "Versions (select with -v):"
                  << "\n";
        for (int i = 0; i < versions.size(); ++i) {
            auto &v = versions[i];
            std::cout << i << ":\t" << v.first << "\n";
        }
        std::cout << std::endl;

        std::cout << "Running " << selected.first << "\n" << std::endl;
    }

    for (int run_idx = 1; run_idx <= num_runs; run_idx++) {
    selected.second(device_csrRowIndices, device_csrColIndices, device_csrVal, num_rows, nnz,
                    matrix_is_zero_indexed, num_devices, iter_max, x_final_result,
                    single_gpu_runtime, compare_to_single_gpu, compare_to_cpu, x_ref_single_gpu,
                    x_ref_cpu);

        // Only compare correctness on first run
        compare_to_single_gpu = false;
        compare_to_cpu = false;
    }

    CUDA_RT_CALL(cudaFree(device_csrRowIndices));
    CUDA_RT_CALL(cudaFree(device_csrColIndices));
    CUDA_RT_CALL(cudaFree(device_csrVal));

    free(host_csrRowIndices);
    free(host_csrColIndices);
    free(host_csrVal);

    if (compare_to_single_gpu || compare_to_cpu) {
        cudaFreeHost(x_final_result);

        if (compare_to_single_gpu) {
            cudaFreeHost(x_ref_single_gpu);
        }

        if (compare_to_cpu) {
            cudaFreeHost(x_ref_cpu);
        }
    }

    MPI_CALL(MPI_Finalize());
}
