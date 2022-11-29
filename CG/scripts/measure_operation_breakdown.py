import subprocess
import re
import os
from os.path import dirname, realpath
from collections import OrderedDict, defaultdict
import sys
from datetime import datetime
import io

import csv

MAX_NUM_GPUS = 2
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
SAVE_NSYS_REPORTS_TO_DIR_PATH = None
NUM_ITERATIONS = 100
EXECUTABLE_NAME = 'cg'
GPU_MODEL = None

VERSION_NAME_TO_IDX_MAP = {
    'Baseline Discrete Standard': 0,
    'Baseline Discrete Pipelined (No Overlap)': 1,
}

MATRIX_NAMES = [
    '(generated)_tridiagonal',
    # 'ecology2',
    #   'shallow_water2', Too little non-zeros
    #   'Trefethen_2000', Too little non-zeros
    'hood',
    'bmwcra_1',
    'consph',
    'thermomech_dM',
    'tmt_sym',
    'crankseg_1',
    'crankseg_2',
]

OPERATION_LABELS_TO_IDX_MAP = OrderedDict(
    [
        ('Dot', 0),
        ('SpMV', 1),
        ('Saxpy', 2),
    ]
)


VERSION_LABELS = VERSION_NAME_TO_IDX_MAP.keys()
GPU_COLUMN_NAMES = None


def get_perf_data_string(result_map, column_labels):
    ephemereal_csv_file = io.StringIO('')

    csv_writer = csv.writer(ephemereal_csv_file, delimiter=',')

    # Add empty string column to get table-like output
    padded_column_labels = [''] + column_labels

    csv_writer.writerow(padded_column_labels)

    for row_label, perf_data in result_map.items():
        final_row = [row_label] + perf_data
        csv_writer.writerow(final_row)

    perf_data_string = ephemereal_csv_file.getvalue()

    return perf_data_string


def parse_nsys_stats_output(stats_string):
    ephemereal_csv_file = io.StringIO(stats_string)

    print(stats_string)

    csv_reader = csv.DictReader(ephemereal_csv_file)

    operation_runtimes = [0] * len(OPERATION_LABELS_TO_IDX_MAP)

    for row in csv_reader:
        operation_label = row['Range']
        operation_runtimes[OPERATION_LABELS_TO_IDX_MAP[operation_label]
                           ] = row['Time (%)']

    return operation_runtimes


def save_results(save_result_to_path, matrix_to_version_to_result_map, version_to_matrix_to_result_map, operation_labels):
    with open(save_result_to_path, 'w') as output_file:
        output_file.write('Results per matrix; rows are versions')
        output_file.write('\n\n')

        for matrix_name, version_to_result_map in matrix_to_version_to_result_map.items():
            output_file.write(f'Results for matrix {matrix_name} =>')
            output_file.write('\n')
            output_file.write(get_perf_data_string(
                version_to_result_map, operation_labels))
            output_file.write('\n\n')

        output_file.write('\n')
        output_file.write('Results per version; rows are matrices')
        output_file.write('\n\n')

        for version_name, matrix_to_result_map in version_to_matrix_to_result_map.items():
            output_file.write(f'Results for version {version_name} =>')
            output_file.write('\n')
            output_file.write(get_perf_data_string(
                matrix_to_result_map, operation_labels))
            output_file.write('\n\n')


def measure_operation_breakdown(save_result_to_path, executable_dir):
    for num_gpus in range(1, MAX_NUM_GPUS + 1):
        cuda_string = CUDA_VISIBLE_DEVICES_SETTING[num_gpus]
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_string

        matrix_to_version_to_result_map = dict.fromkeys(MATRIX_NAMES)
        version_to_matrix_to_result_map = dict.fromkeys(
            VERSION_LABELS)

        for matrix_name in MATRIX_NAMES:
            matrix_path = MATRICES_FOLDER_PATH + '/' + matrix_name + '.mtx'

            if 'generated' in matrix_name:
                matrix_path = None

            version_to_result_map = defaultdict(list)

            for version_name, version_idx in VERSION_NAME_TO_IDX_MAP.items():
                executable_path = executable_dir + '/' + EXECUTABLE_NAME
                full_executable = f'{executable_path} -s 1 -v {version_idx} -niter {NUM_ITERATIONS}'

                if matrix_path:
                    full_executable += f' -matrix_path {matrix_path}'

                nsys_report_output_filename = SAVE_NSYS_REPORTS_TO_DIR_PATH + \
                    f'/report-{num_gpus}{GPU_MODEL}-version{version_idx}-{matrix_name}.nsys-rep'

                nsys_profile_command = 'nsys profile'
                nsys_profile_command += ' '

                # Trace only NVTX ranges
                nsys_profile_command += '--trace nvtx'
                nsys_profile_command += ' '

                nsys_profile_command += \
                    f'--output {nsys_report_output_filename}'
                nsys_profile_command += ' '

                # Overwrite any existing reports
                nsys_profile_command += '--force-overwrite true'
                nsys_profile_command += ' '

                # Pass executable with arguments
                nsys_profile_command += f'{full_executable}'

                subprocess.run(
                    nsys_profile_command.split(), capture_output=False)

                nsys_stat_command = f'nsys stats {nsys_report_output_filename} --report nvtxppsum --format csv --quiet'

                nsys_stats_output = subprocess.run(
                    nsys_stat_command.split(), capture_output=True)

                nsys_stats_output = nsys_stats_output.stdout.decode(
                    'utf-8')

                newline_slices = nsys_stats_output.split('\n')
                nsys_stats_output = '\n'.join([
                    line for line in newline_slices if line.strip() != ''])

                operation_runtimes = parse_nsys_stats_output(
                    nsys_stats_output)

                version_to_result_map[version_name][:] = operation_runtimes[:]

            matrix_to_version_to_result_map[matrix_name] = version_to_result_map

        for version_name in VERSION_LABELS:
            matrix_to_result_map = dict()

            for matrix_name in MATRIX_NAMES:
                result = matrix_to_version_to_result_map[matrix_name][version_name]
                matrix_to_result_map[matrix_name] = result

            version_to_matrix_to_result_map[version_name] = matrix_to_result_map

        per_gpu_result_path = save_result_to_path + f'_{num_gpus}GPU' + '.txt'

        operation_labels = list(OPERATION_LABELS_TO_IDX_MAP.keys())

        save_results(per_gpu_result_path, matrix_to_version_to_result_map,
                     version_to_matrix_to_result_map, operation_labels)


if __name__ == "__main__":
    dir_path = dirname(realpath(__file__))

    SAVE_NSYS_REPORTS_TO_DIR_PATH = dir_path + '/../nsys_reports'
    SAVE_RESULT_TO_DIR_PATH = dir_path + '/../results'
    EXECUTABLE_DIR = dir_path + '/../bin'

    arg_idx = 1

    while arg_idx < len(sys.argv):
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

            MAX_NUM_GPUS = int(sys.argv[arg_idx])

        if sys.argv[arg_idx] == '--gpu_model':
            arg_idx += 1

            GPU_MODEL = sys.argv[arg_idx]

        arg_idx += 1

    BASE_FILENAME = 'cg_operation_breakdown'

    SAVE_RESULT_TO_FILE_PATH = SAVE_RESULT_TO_DIR_PATH + '/' + BASE_FILENAME

    GPU_COLUMN_NAMES = [str(num_gpus) + ' GPU' + ('s' if num_gpus != 1 else '')
                        for num_gpus in range(1, MAX_NUM_GPUS + 1)]

    measure_operation_breakdown(SAVE_RESULT_TO_FILE_PATH, EXECUTABLE_DIR)
