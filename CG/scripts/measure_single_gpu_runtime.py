import subprocess
import re
import os
from os.path import dirname, realpath
from collections import defaultdict
import sys
from datetime import datetime
import csv
import io
import csv
from collections import namedtuple

Version = namedtuple("Version", "version_name version_index")

MATRICES_FOLDER_PATH = '/global/D1/homes/iismayilov/matrices'
NUM_RUNS = 5
NUM_ITERATIONS = 1000
EXECUTABLE_NAME = 'cg'
GPU_MODEL = None
USING_NVSHMEM = True

SINGLE_GPU_VERSION = Version("Single GPU Discrete Standard NVSHMEM", 8)

MATRIX_NAMES = [
    '(generated)_tridiagonal',
    'ecology2',
    #   'shallow_water2', Too little non-zeros
    #   'Trefethen_2000', Too little non-zeros
    'hood',
    'bmwcra_1',
    'consph',
    'thermomech_dM',
    'tmt_sym',
    'crankseg_1',
    'crankseg_2',
    'Queen_4147',
    'Bump_2911',
    'G3_circuit',
    'StocF-1465',
    'Flan_1565',
    'audikw_1',
    'Serena',
    'Geo_1438',
    'Hook_1498',
    #   'bone010', Multi-part matrix, don't handle those for now
    'ldoor'
]

EXECUTION_TIME_REGEX = 'Execution time:\s+(?P<exec_time>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?) s'


def write_to_csv(matrix_to_single_gpu_runtime_map, output_csv_file):
    column_labels = ['Version', 'Runtime']

    with open(output_csv_file, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')

        # Add empty string column to get table-like output
        csv_writer.writerow(column_labels)

        for matrix_name, single_gpu_runtime in matrix_to_single_gpu_runtime_map.items():
            row = [matrix_name, single_gpu_runtime]
            csv_writer.writerow(row)


def measure_runtime(save_result_to_path, executable_dir):
    execution_time_regex_pattern = re.compile(EXECUTION_TIME_REGEX)

    matrix_to_single_gpu_runtime_map = dict.fromkeys(MATRIX_NAMES)

    for matrix_name in MATRIX_NAMES:
        matrix_path = MATRICES_FOLDER_PATH + '/' + matrix_name + '.mtx'

        if 'generated' in matrix_name:
            matrix_path = None

        # Only run on the first GPU
        cuda_string = "0"
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_string

        executable_path = executable_dir + '/' + EXECUTABLE_NAME
        command = f'{executable_path} -s 1 -v {SINGLE_GPU_VERSION.version_index} -niter {NUM_ITERATIONS} -num_runs {NUM_RUNS}'

        if matrix_path:
            command += f' -matrix_path {matrix_path}'

        if USING_NVSHMEM:
            command = f'mpirun -np 1' + ' ' + command

        print(command)

        output = subprocess.run(
            command.split(), capture_output=True)

        output = output.stdout.decode('utf-8')

        runtimes = output.splitlines()

        execution_times = []

        for _run_idx, runtime in enumerate(runtimes):
            execution_time_match = execution_time_regex_pattern.match(
                runtime)
            execution_time_on_run = float(
                execution_time_match.group('exec_time'))

            execution_times.append(execution_time_on_run)

            print(f'Run {_run_idx} took {execution_time_on_run}')

        min_execution_time = min(execution_times)

        matrix_to_single_gpu_runtime_map[matrix_name] = min_execution_time

    write_to_csv(matrix_to_single_gpu_runtime_map, save_result_to_path)


if __name__ == "__main__":
    dir_path = dirname(realpath(__file__))

    SAVE_RESULT_TO_DIR_PATH = dir_path + '/../results'
    EXECUTABLE_DIR = dir_path + '/../bin'
    FILENAME = None
    NUM_GPUS = None

    arg_idx = 1

    while arg_idx < len(sys.argv):
        if sys.argv[arg_idx] == '--filename':
            arg_idx += 1
            arg_val = sys.argv[arg_idx]

            if arg_val != 'USE_DEFAULT_FILENAME':
                FILENAME = arg_val

        if sys.argv[arg_idx] == '--num_iter':
            arg_idx += 1

            NUM_ITERATIONS = sys.argv[arg_idx]

        if sys.argv[arg_idx] == '--matrices_folder':
            arg_idx += 1
            arg_val = sys.argv[arg_idx]

            if arg_val != 'USE_DEFAULT_MATRICES_FOLDER':
                MATRICES_FOLDER_PATH = arg_val

        if sys.argv[arg_idx] == '--gpu_model':
            arg_idx += 1

            GPU_MODEL = sys.argv[arg_idx]

        if sys.argv[arg_idx] == '--num_runs':
            arg_idx += 1

            NUM_RUNS = int(sys.argv[arg_idx])

        if sys.argv[arg_idx] == '-use_nvshmem':
            USING_NVSHMEM = True

        arg_idx += 1

    BASE_FILENAME = 'cg_runtime'

    if USING_NVSHMEM:
        EXECUTABLE_NAME = 'cg_nvshmem'
        BASE_FILENAME = 'cg_nvshmem_runtime_single_gpu'

    if FILENAME == None:
        FILENAME = BASE_FILENAME + '-' + datetime.now().strftime('%d-%m-%Y_%H-%M-%S') + \
            f'-{GPU_MODEL}' + '.csv'

    SAVE_RESULT_TO_FILE_PATH = SAVE_RESULT_TO_DIR_PATH + '/' + FILENAME

    print(SAVE_RESULT_TO_FILE_PATH)

    measure_runtime(SAVE_RESULT_TO_FILE_PATH, EXECUTABLE_DIR)
