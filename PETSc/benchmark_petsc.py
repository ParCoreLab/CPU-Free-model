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
EXECUTABLE_NAME = 'petsc_cg_solver'
GPU_MODEL = 'A100'

VERSION_NAME_TO_IDX_MAP = {
    'PETSc Standard CG': 0,
    'PETSc Pipelined CG': 1,
}

VERSION_INDICES_TO_RUN = list(VERSION_NAME_TO_IDX_MAP.values())
VERSION_LABELS = VERSION_NAME_TO_IDX_MAP.keys()

MATRIX_NAMES = [
    # 'tridiagonal',
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

GPU_COLUMN_NAMES = None

def write_to_csv(matrix_to_version_to_result_map, column_labels, output_csv_file):
    padded_column_labels = ['Matrix', 'Version'] + column_labels

    with open(output_csv_file, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow([f'Execution time on {MAX_NUM_GPUS}{GPU_MODEL}'])
        csv_writer.writerow(padded_column_labels)

        for matrix_name, version_to_result_map in matrix_to_version_to_result_map.items():
            for version_name, runtimes in version_to_result_map.items():
                final_row = [matrix_name, version_name] + runtimes
                csv_writer.writerow(final_row)


def measure_runtime(save_result_to_path, executable_dir):
    matrix_to_version_to_result_map = dict.fromkeys(MATRIX_NAMES)

    filtered_version_name_to_idx_map = dict((version_name, version_idx) for (
        version_name, version_idx) in VERSION_NAME_TO_IDX_MAP.items() if version_idx in VERSION_INDICES_TO_RUN)
    filtered_version_indices = [
        str(version_idx) for version_idx in filtered_version_name_to_idx_map.values()]
    filtered_versions_string = ','.join(filtered_version_indices)

    filtered_version_labels = filtered_version_name_to_idx_map.keys()

    version_to_matrix_to_result_map = dict.fromkeys(
        filtered_version_labels)

    for matrix_name in MATRIX_NAMES:
        matrix_path = MATRICES_FOLDER_PATH + '/' + matrix_name + '.dat'

        version_to_result_map = defaultdict(list)

        for num_gpus in GPU_NUMS_TO_RUN:
            cuda_string = CUDA_VISIBLE_DEVICES_SETTING[num_gpus]
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_string

            executable_path = executable_dir + '/' + EXECUTABLE_NAME
            command = f'{executable_path} -matrix_path {matrix_path}'

            command = f'mpirun -np {num_gpus}' + ' ' + command

            for cur_idx, version_name in enumerate(filtered_version_labels):
                if version_name == 'PETSc Standard CG':
                    command += ' -ksp_type cg'
                elif version_name == 'PETSc Pipelined CG':
                    command += ' -ksp_type pipecg'

                version_name_blurb = '_'.join(word.lower() for word in version_name.split(' '))
                temp_csv_filename = f'{version_name_blurb}_{num_gpus}GPUs_{matrix_name}.csv'

                command += f' -log_view :{temp_csv_filename}:ascii_csv'

                print(
                    f'Running version {version_name} on matrix {matrix_name} with {num_gpus} GPUs')

                output = subprocess.run(
                command.split(), capture_output=False)       

                min_execution_time = None

                with open(temp_csv_filename) as csvfile:
                    reader = csv.DictReader(csvfile)

                    for row in reader:
                        stageName = row["Stage Name"]
                        eventName = row["Event Name"]
                        rank      = int(row["Rank"])

                        if stageName == 'Main Stage' and eventName == 'KSPSolve' and rank == 0:
                            min_execution_time = float(row["Time"])

                print(min_execution_time)

                os.remove(temp_csv_filename)

                version_to_result_map[version_name].append(min_execution_time)

            matrix_to_version_to_result_map[matrix_name] = version_to_result_map

    for version_name in filtered_version_labels:
        matrix_to_result_map = dict()

        for matrix_name in MATRIX_NAMES:
            result = matrix_to_version_to_result_map[matrix_name][version_name]
            matrix_to_result_map[matrix_name] = result

        version_to_matrix_to_result_map[version_name] = matrix_to_result_map

    write_to_csv(matrix_to_version_to_result_map,
                 GPU_COLUMN_NAMES, save_result_to_path)


if __name__ == "__main__":
    dir_path = dirname(realpath(__file__))

    SAVE_RESULT_TO_DIR_PATH = dir_path + '/results'
    EXECUTABLE_DIR = dir_path + '/bin'
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
            MAX_NUM_GPUS = max(GPU_NUMS_TO_RUN)

        if sys.argv[arg_idx] == '--versions_to_run':
            arg_idx += 1

            versions_to_run = sys.argv[arg_idx].split(',')
            versions_to_run = [int(version_index.strip())
                               for version_index in versions_to_run]

            VERSION_INDICES_TO_RUN = versions_to_run[:]

        if sys.argv[arg_idx] == '--gpu_model':
            arg_idx += 1

            GPU_MODEL = sys.argv[arg_idx]

        if sys.argv[arg_idx] == '--num_runs':
            arg_idx += 1

            NUM_RUNS = int(sys.argv[arg_idx])

        arg_idx += 1

    EXECUTABLE_NAME = 'petsc_cg_solver'
    BASE_FILENAME = 'petsc_cg_runtime'

    if FILENAME == None:
        FILENAME = BASE_FILENAME + '-' + datetime.now().strftime('%d-%m-%Y_%H-%M-%S') + \
            f'-{MAX_NUM_GPUS}{GPU_MODEL}' + '.csv'

    GPU_COLUMN_NAMES = [str(num_gpus) + ' GPU' + ('s' if num_gpus != 1 else '')
                        for num_gpus in GPU_NUMS_TO_RUN]

    SAVE_RESULT_TO_FILE_PATH = SAVE_RESULT_TO_DIR_PATH + '/' + FILENAME

    measure_runtime(SAVE_RESULT_TO_FILE_PATH, EXECUTABLE_DIR)
