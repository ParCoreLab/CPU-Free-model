#!/usr/bin/env python3

#SBATCH -J stencil-bench-weak
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -c 16
#SBATCH -A proj16
#SBATCH -p palamut-cuda
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH -o stencil_bench_%j.log

import os
import sys

sys.path.append(os.getcwd())

from pathlib import Path
from datetime import datetime
from itertools import cycle

import bench

BIN = './jacobi'
BIN_3D = './jacobi3d'

BIN_NVSHMEM = './jacobi_nvshmem'
BIN_3D_NVSHMEM = './jacobi3d_nvshmem'

VERSIONS = [
    0,  # Baseline Copy
    1,  # Baseline Overlap
    2,  # Baseline P2P
    3,  # Design 1
    # 4,  # Design 2
    5  # PERKS
]
VERSIONS_NO_COMPUTE = [
    6,  # Baseline Copy
    7,  # Baseline Overlap
    8,  # Baseline P2P
    9,  # Design 1
    10  # Design 2
]

VERSIONS_NVSHMEM = [
    0,  # Baseline
    1,  # Design 1
    # 2,  # Design 2
    3,  # Design 1 Partitioned
    # 3  # PERKS
    7,  # PERKS, possibly
]

VERSIONS_NVSHMEM_NO_COMPUTE = [
    4,  # Baseline
    5,  # Design 1
    6,  # Design 2
    7  # PERKS
]

NUM_REPEAT = 1

BASE_DIR = Path(str(datetime.now()))
BASE_DIR.mkdir()


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
    {'starting_dim': (256, 256), 'num_iter': 1_000_000},
    {'starting_dim': (1024, 1024), 'num_iter': 1_000_000},
    {'starting_dim': (2048, 1024), 'num_iter': 1_000_000},
    {'starting_dim': (8192, 4096), 'num_iter': 10_000},
]

strong_scaling = [
    {'starting_dim': (4096, 4096), 'num_iter': 10000},
]

weak_scaling_3D = [
    {'bin': BIN_3D, 'starting_dim': (256, 256, 256), 'num_iter': 10000},
    {'bin': BIN_3D, 'starting_dim': (256, 256, 256), 'num_iter': 10000, 'dim_func': dim_func_last},
]

strong_scaling_3D = [
    {'bin': BIN_3D, 'starting_dim': (512, 512, 512), 'num_iter': 10000},
]


def run_experiment(name: str, args):
    dim_str = get_dim_str(args['starting_dim'])
    args['out_file'] = BASE_DIR / f'{name}_{dim_str}.csv'
    bench.run(**args)


def run(args, version=''):
    run_experiment(version, {**args, 'versions': VERSIONS, 'bin': BIN})
    # run_experiment(version, {**args, 'versions': VERSIONS_NVSHMEM, 'bin': BIN_NVSHMEM, 'mpi': True})

    # run_experiment(f'{version}_No_Compute', {**args, 'versions': VERSIONS_NO_COMPUTE, 'bin': BIN})
    # run_experiment(f'{version}_No_Compute',
    #                {**args, 'versions': VERSIONS_NVSHMEM_NO_COMPUTE, 'bin': BIN_NVSHMEM, 'mpi': True})


if __name__ == '__main__':
    # Running with the same name merges them
    for args in weak_scaling:
        run({**default_args, **args}, version='2D_Weak_Scaling')

    # for args in weak_scaling_3D:
    #     run({**default_args, **args}, version='3D_Weak_Scaling')

    # for args in strong_scaling:
    #     run({**default_args_strong, **args}, version='2D_Strong_Scaling')

    # for args in strong_scaling_3D:
    #     run({**default_args_strong, **args}, version='3D_Strong_Scaling')
