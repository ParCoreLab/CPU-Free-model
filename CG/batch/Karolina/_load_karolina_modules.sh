ml NVSHMEM/2.9.0-gompi-2022a-CUDA-11.7.0
ml Python/3.9.5-GCCcore-10.3.0
ml PETSc/3.17.4-foss-2021a

export PETSC_DIR=/apps/all/PETSc/3.17.4-foss-2021a
export NVSHMEM_IB_ENABLE_IBGDA=true
export PYTHONPATH=$PETSC_DIR/lib/petsc/bin/:$PYTHONPATH