source /home/iismayilov21/spack/share/spack/setup-env.sh

spack load petsc@3.17.4

export MPI_HOME=$SPACK_ROOT/opt/spack/linux-ubuntu20.04-zen2/gcc-11.1.0/openmpi-4.1.4-mgm6hzztbtf3aae66d3ozy4l3yrrrbrb
export PYTHONPATH=$PETSC_DIR/lib/petsc/bin/:$PYTHONPATH

export CXX=/usr/bin/g++-11
