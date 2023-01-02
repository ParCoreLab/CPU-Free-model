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

MAX_NUM_GPUS = 8
GPU_NUMS_TO_RUN = range(1, MAX_NUM_GPUS + 1)

CUDA_VISIBLE_DEVICES_SETTING = [
    "0",
    "0",
    "0,1",
    "0,1,2",
    "0,1,2,3",
    "0,1,2,3,4",
    "0,1,2,3,4,5",
    "0,1,2,3,4,5,6",
    "0,1,2,3,4,5,6,7",
]

MATRICES_FOLDER_PATH = '/global/D1/homes/iismayilov/matrices'
NUM_RUNS = 5
NUM_ITERATIONS = 1000
EXECUTABLE_NAME = 'cg'
GPU_MODEL = None
USING_NVSHMEM = True

VERSION_NAME_TO_IDX_MAP_NVSHMEM = {
    'Baseline Discrete Standard NVSHMEM': 0,
    'Baseline Discrete Pipelined NVSHMEM': 1,
    'Baseline Persistent Standard NVSHMEM': 2,
    '(Ours) Persistent Pipelined NVSHMEM': 3,
    '(Ours) Persistent Pipelined Multi-Overlap NVSHMEM': 4
}

VERSION_NAME_TO_IDX_MAP = VERSION_NAME_TO_IDX_MAP_NVSHMEM.copy()
VERSIONS_INDICES_TO_RUN = list(VERSION_NAME_TO_IDX_MAP_NVSHMEM.values())

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
VERSION_LABELS = VERSION_NAME_TO_IDX_MAP.keys()
GPU_COLUMN_NAMES = None


def get_perf_data_string(version_to_result_map, column_labels):
    ephemereal_csv_file = io.StringIO('')

    csv_writer = csv.writer(ephemereal_csv_file, delimiter=',')

    # Add empty string column to get table-like output
    padded_column_labels = [''] + column_labels

    csv_writer.writerow(padded_column_labels)

    for row_label, runtimes in version_to_result_map.items():
        final_row = [row_label] + runtimes
        csv_writer.writerow(final_row)

    perf_data_string = ephemereal_csv_file.getvalue()

    return perf_data_string


def measure_runtime(save_result_to_path, executable_dir):
    execution_time_regex_pattern = re.compile(EXECUTION_TIME_REGEX)

    matrix_to_version_to_result_map = dict.fromkeys(MATRIX_NAMES)
    version_to_matrix_to_result_map = dict.fromkeys(
        VERSION_LABELS)

    for matrix_name in MATRIX_NAMES:
        matrix_path = MATRICES_FOLDER_PATH + '/' + matrix_name + '.mtx'

        if 'generated' in matrix_name:
            matrix_path = None

        version_to_result_map = defaultdict(list)

        for version_name, version_idx in VERSION_NAME_TO_IDX_MAP.items():
            if version_idx not in VERSIONS_INDICES_TO_RUN:
                continue

            for num_gpus in GPU_NUMS_TO_RUN:
                print(
                    f'Running version {version_name} on matrix {matrix_name} with {num_gpus} GPUs')

                cuda_string = CUDA_VISIBLE_DEVICES_SETTING[num_gpus]
                os.environ['CUDA_VISIBLE_DEVICES'] = cuda_string

                executable_path = executable_dir + '/' + EXECUTABLE_NAME
                command = f'{executable_path} -s 1 -v {version_idx} -niter {NUM_ITERATIONS}'

                if matrix_path:
                    command += f' -matrix_path {matrix_path}'

                if USING_NVSHMEM:
                    command = f'mpirun -np {num_gpus}' + ' ' + command

                execution_times = []

                for _ in range(NUM_RUNS):
                    output = subprocess.run(
                        command.split(), capture_output=True)

                    output = output.stdout.decode('utf-8')

                    execution_time_match = execution_time_regex_pattern.match(
                        output)
                    execution_time_on_run = float(
                        execution_time_match.group('exec_time'))

                    execution_times.append(execution_time_on_run)

                min_execution_time = min(execution_times)

                version_to_result_map[version_name].append(min_execution_time)

            matrix_to_version_to_result_map[matrix_name] = version_to_result_map

    for version_name in VERSION_LABELS:
        matrix_to_result_map = dict()

        for matrix_name in MATRIX_NAMES:
            result = matrix_to_version_to_result_map[matrix_name][version_name]
            matrix_to_result_map[matrix_name] = result

        version_to_matrix_to_result_map[version_name] = matrix_to_result_map

    with open(save_result_to_path, 'a') as output_file:
        output_file.write('Results per matrix; rows are versions')
        output_file.write('\n\n')

        for matrix_name, version_to_result_map in matrix_to_version_to_result_map.items():
            output_file.write(f'Results for matrix {matrix_name} =>')
            output_file.write('\n')
            output_file.write(get_perf_data_string(
                version_to_result_map, GPU_COLUMN_NAMES))
            output_file.write('\n\n')

        output_file.write('\n')
        output_file.write('Results per version; rows are matrices')
        output_file.write('\n\n')

        for version_name, matrix_to_result_map in version_to_matrix_to_result_map.items():
            output_file.write(f'Results for version {version_name} =>')
            output_file.write('\n')
            output_file.write(get_perf_data_string(
                matrix_to_result_map, GPU_COLUMN_NAMES))
            output_file.write('\n\n')


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

        if sys.argv[arg_idx] == '--num_gpus':
            arg_idx += 1

            gpu_nums_to_run = sys.argv[arg_idx].split(',')
            gpu_nums_to_run = [int(gpu_num.strip())
                               for gpu_num in gpu_nums_to_run]

            GPU_NUMS_TO_RUN = gpu_nums_to_run[:]

        if sys.argv[arg_idx] == '--versions_to_run':
            arg_idx += 1

            versions_to_run = sys.argv[arg_idx].split(',')
            versions_to_run = [int(version_index.strip())
                               for version_index in versions_to_run]

            VERSIONS_INDICES_TO_RUN = versions_to_run[:]

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
        VERSION_NAME_TO_IDX_MAP = VERSION_NAME_TO_IDX_MAP_NVSHMEM.copy()
        EXECUTABLE_NAME = 'cg_nvshmem'
        BASE_FILENAME = 'cg_nvshmem_runtime'

    if FILENAME == None:
        FILENAME = BASE_FILENAME + '-' + datetime.now().strftime('%d-%m-%Y_%H-%M-%S') + \
            f'-{GPU_MODEL}' + '.txt'

    VERSION_LABELS = VERSION_NAME_TO_IDX_MAP.keys()

    GPU_COLUMN_NAMES = [str(num_gpus) + ' GPU' + ('s' if num_gpus != 1 else '')
                        for num_gpus in GPU_NUMS_TO_RUN]

    SAVE_RESULT_TO_FILE_PATH = SAVE_RESULT_TO_DIR_PATH + '/' + FILENAME

    measure_runtime(SAVE_RESULT_TO_FILE_PATH, EXECUTABLE_DIR)
