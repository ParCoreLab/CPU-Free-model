#!/bin/sh

. /truba/home/dsagbili/spack/share/spack/setup-env.sh
spack load nvshmem@2.7.0-6
export LD_LIBRARY_PATH=/truba/home/dsagbili/spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/nvshmem-2.7.0-6-pdl77w7adu5dm334pezvemvt5tjxsowg/lib:/truba/home/dsagbili/spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/ucx-1.13.1-tc7ltbeqjfzr4sdwbv5jgppl4p62q5mu/lib:$LD_LIBRARY_PATH
export UCX_HOME=/truba/home/dsagbili/spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/ucx-1.13.1-tc7ltbeqjfzr4sdwbv5jgppl4p62q5mu
export NVSHMEM_HOME=/truba/home/dsagbili/spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/nvshmem-2.7.0-6-pdl77w7adu5dm334pezvemvt5tjxsowg
export MPI_HOME=/truba/home/dsagbili/spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/openmpi-4.1.4-ycvxffyzzonogvqycd4gpp7aholtkss5
export CUDA_HOME=/truba/home/dsagbili/spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/cuda-11.8.0-37xn6z7age2zvgrmug5jad7l34sizzkp
make -B