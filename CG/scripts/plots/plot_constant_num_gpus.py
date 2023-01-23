
from itertools import cycle
from os.path import dirname, realpath

from common import get_files, markers, get_module_dir, wrap_labels

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

MODULE_DIR = get_module_dir('Constant Number of GPUs')

dir_path = dirname(realpath(__file__))

# plt.style.use(dir_path + '/default.mplstyle')
plt.style.use(dir_path + '/paper.mplstyle')
# plt.style.use('fivethirtyeight')

files = get_files()

for file in files:
    title = file.readline().strip()

    data = pd.read_csv(file, index_col='Matrix')
    data = data.sort_index()

    matrix_names = np.unique(
        [matrix_name for matrix_name, _ in data.iterrows()])

    gpu_num_column_labels = [column_label
                             for column_label in data.columns if 'GPU' in column_label]

    for matrix_name in matrix_names:
        if matrix_name not in MATRIX_NAMES:
            data.drop(matrix_name, inplace=True)

    for gpu_num_column_label in gpu_num_column_labels:
        per_gpu_num_data = data[['Version', gpu_num_column_label]]
        per_gpu_num_data = per_gpu_num_data.pivot_table(
            gpu_num_column_label, 'Matrix', 'Version')

        ax = per_gpu_num_data.plot.bar()

        ax.set_ylabel('Time')
        ax.set_title('Execution time per matrix')
        ax.legend()
        # ax.legend(ax.get_lines(), per_gpu_num_data.columns, loc='best')

        # per_gpu_num_title = f'{title} ({matrix_name})'
        per_gpu_num_title = gpu_num_column_label

        plt.title(per_gpu_num_title)
        plt.savefig(MODULE_DIR / per_gpu_num_title)

    plt.show()
