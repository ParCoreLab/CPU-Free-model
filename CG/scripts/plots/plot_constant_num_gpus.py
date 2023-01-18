
from itertools import cycle
from os.path import dirname, realpath

from common import get_files, markers, get_module_dir, wrap_labels

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

MATRIX_NAMES = [
    # '(generated)_tridiagonal',
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

    label_locations = np.arange(len(MATRIX_NAMES))
    bar_width = 0.8

    versions_grouped_data = data.groupby(
        'Version').agg(lambda x: list(x))
    num_versions = len(versions_grouped_data.index)
    per_bar_width = bar_width / num_versions

    for gpu_num_column_label in gpu_num_column_labels:
        per_gpu_num_data = versions_grouped_data[gpu_num_column_label]
        per_gpu_num_data = per_gpu_num_data.sort_index()

        fig, ax = plt.subplots()

        bars = []

        for idx, (version_label, per_matrix_result) in enumerate(per_gpu_num_data.items()):
            x_offset = (idx - len(MATRIX_NAMES) / 2) * \
                bar_width + bar_width / 2

            tmp_bar = ax.bar(label_locations - bar_width / 2 +
                             idx * per_bar_width, per_matrix_result, bar_width, label=version_label)

            bars.append(tmp_bar)

        ax.set_ylabel('Time')
        ax.set_title('Execution time per matrix')
        ax.set_xticks(label_locations, MATRIX_NAMES)
        ax.legend()

        for tmp_bar in bars:
            ax.bar_label(tmp_bar, padding=3)

        # fig.tight_layout()

    plt.show()

    #     axes = per_matrix_data.plot()

    #     markers_cycle = cycle(markers)

    #     for line in axes.get_lines():
    #         line.set_marker(next(markers_cycle))

    #         # If baseline version
    #         if line.get_label().lower().startswith('baseline'):
    #             line.set_linestyle('dashed')

    #         markers_cycle = cycle(markers)

    #     axes.legend(axes.get_lines(), per_matrix_data.columns, loc='best')

    #     # per_matrix_title = f'{title} ({matrix_name})'
    #     per_matrix_title = matrix_name

    #     plt.title(per_matrix_title)
    #     plt.savefig(MODULE_DIR / per_matrix_title)

    # plt.show()
