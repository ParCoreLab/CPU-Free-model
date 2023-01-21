#!/usr/bin/env python3

#SBATCH --job-name=stencil-bench
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --partition hgx2q
#SBATCH --time=04:00:00
#SBATCH --output=sbatch_output_%j.log

import os
import re
import subprocess
from itertools import accumulate, cycle

import pandas as pd

##########################################
# Config
BIN = ['./jacobi']
PRE_ARGS = ['-s']           # No header output

PRINT = False               # For debugging

VERSIONS = list(range(14))  # Version numbers to run
NUM_REPEAT = 10000          # Number of times to repeat the experiments
REPEAT_REDUCE = min         # Function to reduce repetitions to a single number
NUM_ITER = 5
STARTING_DIM = [1024, 1024]
GPU_STEP = lambda x: x * 2  # How the next GPU count is calculated. Doubled by default

OUT_FILE = '/dev/stdout'    # File to write csv to


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


def _gpu_setting_generator(settings, max=_num_gpus):
    index = 1

    while index <= max:
        yield settings[index - 1]
        index = GPU_STEP(index)


CUDA_VISIBLE_DEVICES_SETTING = list(accumulate(map(str, range(0, _num_gpus)), func=lambda a, b: f'{a},{b}'))
GPU_INDICES = list(_gpu_setting_generator(CUDA_VISIBLE_DEVICES_SETTING, max=_num_gpus))

# 0:	Baseline Multi Threaded Copy etc
VERSION_REGEX = re.compile(r'(\d+):\s+(.*)', re.MULTILINE)


def dim_to_dim(dims):
    alphabet = (f'-n{a}' for a in ['x', 'y', 'z'])
    return [item for t in zip(alphabet, dims) for item in t]


def run_execution_time(args: []):
    out = subprocess.run(args, capture_output=True).stdout.decode()
    return float(re.findall(r'\d+\.\d+', out)[0])


if __name__ == '__main__':
    # Get actual version names
    full_out = subprocess.run(BIN, capture_output=True).stdout.decode()
    version_names = [name for _, name in re.findall(VERSION_REGEX, full_out)]

    # Pre-compute dimensions for each GPU count
    dim_generator = DIM_FUNC(STARTING_DIM)
    dims = [next(dim_generator) for _ in range(len(GPU_INDICES))]

    # 1 GPUs (256x256), 2 GPUs (256x512) ...
    columns = list(map(lambda dims: f'{dims[0] + 1} GPUs ({"x".join([str(x) for x in dims[1]])})', enumerate(dims)))

    # Create output csv handler
    results = pd.DataFrame(index=[*version_names[:len(VERSIONS)]], columns=columns)
    results = results.rename_axis('Version')

    for v, name in zip(VERSIONS, version_names):
        for i, (dim, gpu_setting) in enumerate(zip(dims, GPU_INDICES)):
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_setting

            # Get dimensions in -nx x form
            dim = dim_to_dim(dim)

            # Make sure all the args are string
            args = list(map(str, [*BIN, *PRE_ARGS, '-v', v, '-niter', NUM_ITER, *dim]))

            if PRINT:
                num_gpus = gpu_setting.count(',')
                print(f'Running {name} on {num_gpus + 1} GPUs {args} ->', end=' ', flush=True)

            execution_time = REPEAT_REDUCE([run_execution_time(args) for _ in range(NUM_REPEAT)])

            if PRINT:
                print(f'{execution_time} seconds')

            results.loc[name].iloc[i] = execution_time

    results.to_csv(OUT_FILE)

