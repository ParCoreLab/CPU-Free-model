#!/bin/sh

module load nvidia/nvhpc/22.3
module load gcc/11.2.0
module load python-pandas-1.0.3

COMM_LIBS_PATH="$NVHPC_ROOT"/comm_libs

export NVCC="$NVHPC_ROOT"/cuda/bin/nvcc
export MPI_HOME="$COMM_LIBS_PATH"/mpi
export NVSHMEM_HOME="$COMM_LIBS_PATH"/nvshmem
