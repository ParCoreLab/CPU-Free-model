
import numpy as np
import pandas as pd
from os.path import dirname, realpath
import argparse

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

VERSIONS_TO_KEEP = [
    '(Baseline) Discrete Standard',
    '(Baseline) Discrete Pipelined',
    '(Ours) Persistent Standard',
    '(Ours) Persistent Pipelined'
]

dir_path = dirname(realpath(__file__))

# First file should be the full CSV file
# Second should be the SingleGPU runtimes
parser = argparse.ArgumentParser()
parser.add_argument('files', type=argparse.FileType('r'), nargs='+')
files = parser.parse_args().files

full_runtimes_csv = files[0]
single_gpu_runtimes_csv = files[1]

if __name__ == "__main__":
    dir_path = dirname(realpath(__file__))

    SAVE_RESULT_TO_DIR_PATH = dir_path + '/../results'

    # Skip first line
    full_runtimes_csv.readline()

    data = pd.read_csv(full_runtimes_csv, index_col='Matrix')
    data = data.sort_index()

    single_gpu_baseline_standard_runtimes = pd.read_csv(
        single_gpu_runtimes_csv, index_col='Matrix')['Runtime']
    single_gpu_baseline_standard_runtimes = single_gpu_baseline_standard_runtimes.sort_index()

    matrix_names = np.unique(
        [matrix_name for matrix_name, _ in data.iterrows()])

    gpu_num_column_labels = [column_label
                             for column_label in data.columns if 'GPU' in column_label]

    for matrix_name in matrix_names:
        if matrix_name not in MATRIX_NAMES:
            data.drop(matrix_name, inplace=True)
            single_gpu_baseline_standard_runtimes.drop(
                matrix_name, inplace=True)

    for gpu_num_column_label in gpu_num_column_labels:
        per_gpu_num_data = data[['Version', gpu_num_column_label]]
        per_gpu_num_data = per_gpu_num_data.pivot_table(
            gpu_num_column_label, 'Matrix', 'Version')

        per_gpu_num_data = pd.DataFrame(
            per_gpu_num_data, columns=VERSIONS_TO_KEEP)

        per_gpu_num_speedup = 1 / per_gpu_num_data.div(
            single_gpu_baseline_standard_runtimes, axis=0)

        pipelined_cg_speedup = per_gpu_num_speedup['(Ours) Persistent Pipelined'] / \
            per_gpu_num_speedup['(Baseline) Discrete Pipelined']
        standard_cg_speedup = per_gpu_num_speedup['(Ours) Persistent Standard'] / \
            per_gpu_num_speedup['(Baseline) Discrete Standard']

        pipelined_cg_geo_mean_spedup = np.exp(
            np.log(pipelined_cg_speedup).mean())
        standard_cg_geo_mean_spedup = np.exp(
            np.log(standard_cg_speedup).mean())

        pipelined_speedup_file_path = SAVE_RESULT_TO_DIR_PATH + \
            '/pipelined_speedup/pipelined_cg_speedup_' + gpu_num_column_label + '.txt'
        standard_speedup_file_path = SAVE_RESULT_TO_DIR_PATH + \
            '/standard_speedup/pipelined_cg_speedup_' + gpu_num_column_label + '.txt'

        with open(pipelined_speedup_file_path, 'w') as pipelined_speedup_file:
            pipelined_cg_speedup.to_string(
                pipelined_speedup_file, header=False)
            pipelined_speedup_file.write('\n')
            pipelined_speedup_file.write(
                f'Persistent vs Discrete Pipelined CG geo mean speedup on {gpu_num_column_label}: {pipelined_cg_geo_mean_spedup}')

        with open(standard_speedup_file_path, 'w') as standard_speedup_file:
            standard_cg_speedup.to_string(
                standard_speedup_file, header=False)
            standard_speedup_file.write('\n')
            standard_speedup_file.write(
                f'Persistent vs Discrete Standard CG geo mean speedup on {gpu_num_column_label}: {standard_cg_geo_mean_spedup}')
