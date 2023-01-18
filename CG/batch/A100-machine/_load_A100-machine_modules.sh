source /home/iismayilov21/spack/share/spack/setup-env.sh

spack load nvshmem@2.7.0-6

export UCX_WARN_UNUSED_ENV_VARS=n
export UCX_HOME=$SPACK_ROOT/opt/spack/linux-ubuntu20.04-zen2/gcc-11.1.0/ucx-1.13.1-cv37hs5p3lpknxhmuhucbsjotdn653vn/
export NVSHMEM_HOME=$SPACK_ROOT/opt/spack/linux-ubuntu20.04-zen2/gcc-11.1.0/nvshmem-2.7.0-6-svccom42hd6t6fmfru3txongtfpvuynm/
export MPI_HOME=$SPACK_ROOT/opt/spack/linux-ubuntu20.04-zen2/gcc-11.1.0/openmpi-4.1.4-cgf2kyjuumewmbove7jagikdbpo42s6q/
export CUDA_HOME=$SPACK_ROOT/opt/spack/linux-ubuntu20.04-zen2/gcc-11.1.0/cuda-11.8.0-vb4kpzvmja7a3pinvxpbschaqo4jkalp/
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$UCX_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CXX=/usr/bin/g++-11
