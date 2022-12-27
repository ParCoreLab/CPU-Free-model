import subprocess
import re
import os
from os.path import dirname, realpath
from collections import OrderedDict, defaultdict
import sys
from datetime import datetime
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
SAVE_NSYS_REPORTS_TO_DIR_PATH = None
NUM_ITERATIONS = 5000
EXECUTABLE_NAME = 'cg_nvtx'
GPU_MODEL = None
USING_NVSHMEM = False

VERSION_NAME_TO_IDX_MAP_REGULAR = {
    'Baseline Discrete Standard': 4,
    'Baseline Discrete Pipelined (No Overlap)': 5,
}

VERSION_NAME_TO_IDX_MAP_NVSHMEM = {
    'Baseline Discrete Standard NVSHMEM': 4,
    'Baseline Discrete Pipelined NVSHMEM (No Overlap)': 5,
}

VERSION_NAME_TO_IDX_MAP = VERSION_NAME_TO_IDX_MAP_REGULAR.copy()

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
]

VERSION_LABELS = None
GPU_COLUMN_NAMES = None


def get_perf_data_string(version_to_matrix_to_result_map, column_labels):
    ephemereal_csv_file = io.StringIO('')

    csv_writer = csv.writer(ephemereal_csv_file, delimiter=',')

    # Add empty string column to get table-like output
    padded_column_labels = [''] + column_labels

    csv_writer.writerow(padded_column_labels)

    for row_label, runtimes in version_to_matrix_to_result_map.items():
        final_row = [row_label] + runtimes
        csv_writer.writerow(final_row)

    perf_data_string = ephemereal_csv_file.getvalue()

    return perf_data_string


def parse_nsys_stats_output(stats_string):
    ephemereal_csv_file = io.StringIO(stats_string)

    csv_reader = csv.DictReader(ephemereal_csv_file)

    labels = []
    raw_runtimes = []

    for row in csv_reader:
        labels.append(row['Range'])
        raw_runtimes.append(row['Total Time (sec)'])

    sorted_labels, raw_runtimes = (
        list(t) for t in zip(*sorted(zip(labels, raw_runtimes))))

    return sorted_labels, raw_runtimes


def save_results(save_result_to_path, version_to_matrix_to_result_map, version_to_operation_labels_map):
    with open(save_result_to_path, 'w') as output_file:
        output_file.write('Results per version; rows are matrices')
        output_file.write('\n\n')

        for version_name, matrix_to_result_map in version_to_matrix_to_result_map.items():
            operation_labels = version_to_operation_labels_map[version_name]

            output_file.write(f'Results for version {version_name} =>')
            output_file.write('\n')
            output_file.write(get_perf_data_string(
                matrix_to_result_map, operation_labels))
            output_file.write('\n\n')


def measure_operation_breakdown(save_result_to_path, executable_dir):
    for num_gpus in GPU_NUMS_TO_RUN:
        cuda_string = CUDA_VISIBLE_DEVICES_SETTING[num_gpus]
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_string

        version_to_matrix_to_result_map = dict.fromkeys(
            VERSION_LABELS)
        version_to_operation_labels_map = dict.fromkeys(VERSION_LABELS)

        for version_name, version_idx in VERSION_NAME_TO_IDX_MAP.items():
            matrix_to_result_map = defaultdict(list)
            operation_labels = None

            for matrix_name in MATRIX_NAMES:
                matrix_path = MATRICES_FOLDER_PATH + '/' + matrix_name + '.mtx'

                if 'generated' in matrix_name:
                    matrix_path = None

                executable_path = executable_dir + '/' + EXECUTABLE_NAME
                full_executable = f'{executable_path} -s 1 -v {version_idx} -niter {NUM_ITERATIONS}'

                if matrix_path:
                    full_executable += f' -matrix_path {matrix_path}'

                nsys_report_output_filename = SAVE_NSYS_REPORTS_TO_DIR_PATH + \
                    f'/report-{num_gpus}{GPU_MODEL}-version{version_idx}-{matrix_name}'

                if USING_NVSHMEM:
                    nsys_report_output_filename += '-NVSHMEM'
                    nsys_report_output_filename += '.nsys-rep'

                nsys_profile_command = 'nsys profile'
                nsys_profile_command += ' '

                if USING_NVSHMEM:
                    nsys_profile_command = f'mpirun -np {num_gpus}' + \
                        ' ' + nsys_profile_command

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

                nsys_stat_command = f'nsys stats --report nvtxppsum --format csv --timeunit seconds --quiet {nsys_report_output_filename}'

                nsys_stats_output = subprocess.run(
                    nsys_stat_command.split(), capture_output=True)

                nsys_stats_output = nsys_stats_output.stdout.decode(
                    'utf-8')

                newline_slices = nsys_stats_output.split('\n')
                nsys_stats_output = '\n'.join([
                    line for line in newline_slices if line.strip() != ''])

                sorted_labels, raw_runtimes = parse_nsys_stats_output(
                    nsys_stats_output)

                if not operation_labels:
                    operation_labels = sorted_labels

                matrix_to_result_map[matrix_name][:] = raw_runtimes[:]

            version_to_matrix_to_result_map[version_name] = matrix_to_result_map
            version_to_operation_labels_map[version_name] = operation_labels

        per_gpu_result_path = save_result_to_path + \
            f'_{num_gpus}{GPU_MODEL}' + '.txt'

        save_results(per_gpu_result_path, version_to_matrix_to_result_map,
                     version_to_operation_labels_map)


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

            gpu_nums_to_run = sys.argv[arg_idx].split(',')
            gpu_nums_to_run = [int(gpu_num.strip())
                               for gpu_num in gpu_nums_to_run]

            GPU_NUMS_TO_RUN = gpu_nums_to_run[:]

        if sys.argv[arg_idx] == '--gpu_model':
            arg_idx += 1

            GPU_MODEL = sys.argv[arg_idx]

        if sys.argv[arg_idx] == '-use_nvshmem':
            USING_NVSHMEM = True

        arg_idx += 1

    BASE_FILENAME = 'cg_operation_breakdown'
    SAVE_RESULT_TO_FILE_PATH = SAVE_RESULT_TO_DIR_PATH + '/' + BASE_FILENAME

    if USING_NVSHMEM:
        VERSION_NAME_TO_IDX_MAP = VERSION_NAME_TO_IDX_MAP_NVSHMEM.copy()
        EXECUTABLE_NAME = 'cg_nvshmem_nvtx'

    VERSION_LABELS = VERSION_NAME_TO_IDX_MAP.keys()

    GPU_COLUMN_NAMES = [str(num_gpus) + ' GPU' + ('s' if num_gpus != 1 else '')
                        for num_gpus in range(1, MAX_NUM_GPUS + 1)]

    measure_operation_breakdown(SAVE_RESULT_TO_FILE_PATH, EXECUTABLE_DIR)
