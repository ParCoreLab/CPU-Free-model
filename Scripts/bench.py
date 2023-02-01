#!/usr/bin/env python3

#SBATCH --job-name=stencil-bench
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --partition hgx2q
#SBATCH --time=01:00:00
#SBATCH --output=sbatch_output_%j.log

import os
import re
import subprocess
import sys
from functools import reduce
from itertools import accumulate, cycle
from pathlib import Path

import pandas as pd

##########################################
# Default Config
BIN_MPI = ['mpirun', '-np', '1', '--timeout', str(5 * 60)]  # Timeout after 5 minutes
BIN = ['./jacobi']
PRE_ARGS = ['-s']  # No header output
GIGA = 10**6

LOG = True  # For debugging

VERSIONS = None  # Version numbers to run. None means all
NUM_REPEAT = 5  # Number of times to repeat the experiments
REPEAT_REDUCE = min  # Function to reduce repetitions to a single number
NUM_ITER = 10000
STARTING_DIM = (1024, 1024)
GPU_STEP = lambda x: x * 2  # How the next GPU count is calculated. Doubled by default

OUT_FILE = '/dev/stdout'  # File to write csv to


# Defines how the next dimension is calculated
def DIM_FUNC(dims):
    for i, _ in cycle(reversed(list(enumerate(dims)))):
        yield dims.copy()
        dims[i] *= 2


##########################################

visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
if visible_devices:
    _num_gpus = visible_devices.count(',') + 1
else:
    _num_gpus = int(subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, capture_output=True)
                    .stdout
                    .decode()
                    .count('\n'))


def _gpu_setting_generator(settings, step=GPU_STEP, max=_num_gpus):
    index = 1

    while index <= max:
        yield settings[index - 1]
        index = step(index)


CUDA_VISIBLE_DEVICES_SETTING = list(accumulate(map(str, range(0, _num_gpus)), func=lambda a, b: f'{a},{b}'))
GPU_INDICES = list(_gpu_setting_generator(CUDA_VISIBLE_DEVICES_SETTING, max=_num_gpus))

# 0:	Baseline Multi Threaded Copy etc
VERSION_REGEX = re.compile(r'(\d+):\s+(.*)', re.MULTILINE)


def dim_to_dim(dims):
    alphabet = (f'-n{a}' for a in ['x', 'y', 'z'])
    return [item for t in zip(alphabet, dims) for item in t]


def get_dim_str(dim):
    return 'x'.join([str(x) for x in dim])


def run_execution_time(args: []):
    out = subprocess.run(args, capture_output=True).stdout.decode()
    # We avoid killing the benchmark when one run fails
    return float(next(iter(re.findall(r'\d+\.\d+', out)), 'nan'))


def run(*, bin=BIN, versions=VERSIONS, starting_dim=STARTING_DIM, num_iter=NUM_ITER, dim_func=DIM_FUNC,
        out_file=OUT_FILE, pre_args=PRE_ARGS,
        gpu_step=GPU_STEP, num_repeat=NUM_REPEAT, repeat_reduce=REPEAT_REDUCE, log=LOG, mpi=False):

    gcell_dir = Path(out_file).parent / 'gcell'
    gcell_dir.mkdir(exist_ok=True)

    out_file_gcell = gcell_dir / Path(out_file).name

    # Make sure it's a mutable list
    starting_dim = list(starting_dim)

    gpu_indices = list(_gpu_setting_generator(CUDA_VISIBLE_DEVICES_SETTING, step=gpu_step, max=_num_gpus))

    # Make sure the binary is in a list
    if bin.__class__ == str:
        bin = [bin]

    # Add binary to mpirun
    if mpi:
        bin = BIN_MPI + bin

    # Get actual version names
    full_out = subprocess.run(bin, capture_output=True, check=True).stdout.decode()
    version_names = [name for _, name in re.findall(VERSION_REGEX, full_out)]

    if not versions:
        versions = list(range(len(version_names)))

    # Pre-compute dimensions for each GPU count
    dim_generator = dim_func(starting_dim)
    dims = [next(dim_generator) for _ in range(len(gpu_indices))]

    # 1 GPUs (256x256), 2 GPUs (256x512) ...
    columns = list(map(lambda dims: f'{gpu_indices[dims[0]].count(",") + 1} GPUs ({get_dim_str(dims[1])})', enumerate(dims)))

    # Create output csv handler
    results = pd.DataFrame(index=[version_names[v] for v in versions], columns=columns)
    results = results.rename_axis('Version')

    results_gcell = results.copy(deep=True)

    for v in versions:
        name = version_names[v]

        for i, (dim, gpu_setting) in enumerate(zip(dims, gpu_indices)):
            # I know this is stupid
            num_gpus = gpu_setting.count(',') + 1
            if mpi:
                # This corresponds to -np x
                bin[2] = num_gpus
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_setting

            # Get dimensions in -nx x form
            dim_arg = dim_to_dim(dim)

            # Make sure all the args are string
            args = list(map(str, [*bin, *pre_args, '-v', v, '-niter', num_iter, *dim_arg]))

            if log:
                print(f'{name} on {num_gpus} GPUs {" ".join(args)} ->', end=' ', flush=True, file=sys.stderr)

            execution_sec = repeat_reduce([run_execution_time(args) for _ in range(num_repeat)])
            execution_gcell = (reduce(lambda x, y: x * y, dim) * num_iter) / execution_sec / GIGA

            if log:
                print(f'{execution_sec} seconds ({execution_gcell} GCELLs/s)', file=sys.stderr)

            results.loc[name].iloc[i] = execution_sec
            results_gcell.loc[name].iloc[i] = execution_gcell

    results.to_csv(out_file, mode='a', header=not os.path.exists(out_file))
    results_gcell.to_csv(out_file_gcell, mode='a', header=not os.path.exists(out_file_gcell))


if __name__ == '__main__':
    run()
