#!/bin/sh

module load nvidia/nvhpc/22.3
module load gcc/11.2.0
module load python-3.7.4

COMM_LIBS_PATH="$NVHPC_ROOT"/comm_libs
MATH_LIBS_PATH="$NVHPC_ROOT"/math_libs/lib64

export NVCC="$NVHPC_ROOT"/cuda/bin/nvcc
export MPI_HOME="$COMM_LIBS_PATH"/mpi
export NVSHMEM_HOME="$COMM_LIBS_PATH"/nvshmem
export MATH_LIBS_PATH="$MATH_LIBS_PATH"