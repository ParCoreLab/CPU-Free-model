. /truba/home/dsagbili/spack/share/spack/setup-env.sh

spack load nvshmem@2.7.0-6

export UCX_WARN_UNUSED_ENV_VARS=n
#export NVSHMEM_IB_ENABLE_GPUINITIATED=1
export UCX_HOME=/truba/home/dsagbili/spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/ucx-1.13.1-tc7ltbeqjfzr4sdwbv5jgppl4p62q5mu
export NVSHMEM_HOME=/truba/home/dsagbili/spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/nvshmem-2.7.0-6-pdl77w7adu5dm334pezvemvt5tjxsowg
export MPI_HOME=/truba/home/dsagbili/spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/openmpi-4.1.4-ycvxffyzzonogvqycd4gpp7aholtkss5
export CUDA_HOME=/truba/home/dsagbili/spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/cuda-11.8.0-37xn6z7age2zvgrmug5jad7l34sizzkp
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$UCX_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH