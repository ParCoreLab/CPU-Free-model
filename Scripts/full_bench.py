#!/usr/bin/env python3

# SBATCH --job-name=stencil-bench
# SBATCH --ntasks=8
# SBATCH --gres=gpu:8
# SBATCH --partition hgx2q
# SBATCH --time=04:00:00
# SBATCH --output=sbatch_output_%j.log

import time
from itertools import cycle

import bench

BIN = './jacobi'
BIN_3D = './jacobi3d'

BIN_NVSHMEM = './jacobi_nvshmem'
BIN_3D_NVSHMEM = './jacobi3d_nvshmem'

NUM_REPEAT = 5


def get_timestamp():
    return round(time.time() * 1000)


def get_dim_str(dim):
    return 'x'.join([str(x) for x in dim])


# Multiplies the last index by 2
def dim_func_last(dims):
    last_index = len(dims) - 1

    while True:
        yield dims.copy()
        dims[last_index] *= 2


default_args = {'bin': BIN, 'num_repeat': NUM_REPEAT}
default_args_strong = {
    **default_args,
    'gpu_step': lambda x: x + 1,  # Add 1 more GPU
    'dim_func': lambda x: cycle([x])
}

weak_scaling = [
    {'starting_dim': [256, 256], 'num_iter': 100000},
    {'starting_dim': [1024, 1024], 'num_iter': 100000},
    {'starting_dim': [8192, 4096], 'num_iter': 10000},
]

strong_scaling = [
    {'starting_dim': [4096, 4096], 'num_iter': 10000},
]

weak_scaling_3D = [
    {'bin': BIN_3D, 'starting_dim': [256, 256, 256], 'num_iter': 10000},
    {'bin': BIN_3D, 'starting_dim': [256, 256, 256], 'num_iter': 10000, 'dim_func': dim_func_last},
]

strong_scaling_3D = [
    {'bin': BIN_3D, 'starting_dim': [512, 512, 512], 'num_iter': 10000},
]


def run_experiment(name: str, args):
    timestamp = get_timestamp()
    dim_str = get_dim_str(args['starting_dim'])
    args['out_file'] = f'{name}_{dim_str}_{timestamp}.csv'
    bench.run(**args)


def run():
    [run_experiment('Weak_scaling', {**default_args, **args}) for args in weak_scaling]
    [run_experiment('Strong_scaling', {**default_args_strong, **args}) for args in strong_scaling]
    [run_experiment('Weak_scaling', {**default_args, **args}) for args in weak_scaling_3D]
    [run_experiment('Strong_scaling', {**default_args_strong, **args}) for args in strong_scaling_3D]


if __name__ == '__main__':
    # Regular
    run()

    # NVSHMEM
    default_args['bin'] = BIN_NVSHMEM
    default_args_strong['bin'] = BIN_3D_NVSHMEM

    default_args['mpi'] = True
    default_args_strong['mpi'] = True

    run()
