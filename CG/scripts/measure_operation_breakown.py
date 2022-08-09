import subprocess
import re
from os.path import dirname, realpath
from collections import OrderedDict, defaultdict
import sys
from datetime import datetime

import pandas as pd

MATRICES_BASE_PATH = '/global/D1/homes/iismayilov/matrices'

NUM_RUNS = 5
NUM_ITERATIONS = 10000

EXECUTION_TIME_REGEX = 'Execution time:\s+(?P<exec_time>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?) s'

EXECUTABLE_NAME_TO_STEM_MAP = OrderedDict(
    [
        ('Dot product', 'only-dot-cg'),
        ('SpMV', 'only-spmv-cg'),
        ('Saxpy', 'only-saxpy-cg'),
        # ('Total', 'full-cg'),
    ]
)

VERSION_NAME_TO_IDX_MAP = {
    'Baseline Persistent Kernel with Unified Memory': 0,
    'Baseline Persistent Kernel with Unified Memory (Input vector gathered before SpMV)': 1,
}

MATRIX_NAMES = [
    '(generated) tridiagonal',
    'ecology2',
    'shallow_water2',
    'Trefethen_2000',
    'hood',
    'bmwcra_1',
    'consph',
]

OPERATION_LABELS = EXECUTABLE_NAME_TO_STEM_MAP.keys()
VERSION_LABELS = VERSION_NAME_TO_IDX_MAP.keys()


def get_perf_data_string(version_to_result_map, column_labels):
    row_labels = version_to_result_map.keys()
    full_perf_data = version_to_result_map.values()

    df = pd.DataFrame(full_perf_data, columns=column_labels,
                      index=row_labels)

    perf_data_string = df.to_csv()

    return perf_data_string


def evaluate_operation_breakdown(save_result_to_path, executable_dir):
    execution_time_regex_pattern = re.compile(EXECUTION_TIME_REGEX)

    matrix_to_version_to_result_map = dict.fromkeys(MATRIX_NAMES)
    version_to_matrix_to_result_map = dict.fromkeys(
        VERSION_LABELS)

    for matrix_name in MATRIX_NAMES:
        matrix_path = MATRICES_BASE_PATH + '/' + matrix_name + '.mtx'

        if 'generated' in matrix_name:
            matrix_path = None

        version_to_result_map = defaultdict(list)

        for version_name, version_idx in VERSION_NAME_TO_IDX_MAP.items():
            for _operation_name, operation_executable_stem in EXECUTABLE_NAME_TO_STEM_MAP.items():
                operation_executable = executable_dir + '/' + operation_executable_stem
                command = f'{operation_executable} -s 1 -v {version_idx} -niter {NUM_ITERATIONS}'

                if matrix_path:
                    command += f' -matrix_path {matrix_path}'

                print(command)

                execution_times = []

                for _ in range(NUM_RUNS):
                    output = subprocess.run(
                        command.split(), capture_output=True)

                    print(output)

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
                version_to_result_map, OPERATION_LABELS))
            output_file.write('\n\n')

        output_file.write('\n')
        output_file.write('Results per version; rows are matrices')
        output_file.write('\n\n')

        for version_name, matrix_to_result_map in version_to_matrix_to_result_map.items():
            output_file.write(f'Results for version {version_name} =>')
            output_file.write('\n')
            output_file.write(get_perf_data_string(
                matrix_to_result_map, OPERATION_LABELS))
            output_file.write('\n\n')


if __name__ == "__main__":
    dir_path = dirname(realpath(__file__))

    SAVE_RESULT_TO_DIR_PATH = dir_path + '/../results'
    EXECUTABLE_DIR = dir_path + '/../bin'
    FILENAME = None

    arg_idx = 1

    while arg_idx < len(sys.argv):
        if sys.argv[arg_idx] == '-f':
            arg_idx += 1

            FILENAME = sys.argv[arg_idx]

        if sys.argv[arg_idx] == '--num-iter':
            arg_idx += 1

            NUM_ITERATIONS = sys.argv[arg_idx]

        arg_idx += 1

    if FILENAME == None:
        FILENAME = 'cg_operation_breakdown-' + \
            datetime.now().strftime('%d-%m-%Y_%H-%M-%S') + '.txt'

    SAVE_RESULT_TO_FILE_PATH = SAVE_RESULT_TO_DIR_PATH + '/' + FILENAME

    evaluate_operation_breakdown(SAVE_RESULT_TO_FILE_PATH, EXECUTABLE_DIR)
