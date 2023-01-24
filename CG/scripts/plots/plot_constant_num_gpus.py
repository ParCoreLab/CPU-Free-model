
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from os.path import dirname, realpath

from common import get_files, markers, get_module_dir, wrap_labels

import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.3

MATRIX_NAMES = [
    'tridiagonal',
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

MODULE_DIR = get_module_dir('Constant Number of GPUs')

dir_path = dirname(realpath(__file__))

plt.style.use(
    'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')

colors = [
    '#c6c9cb', '#64b8e5', '#ee7fb2'
]

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

        per_gpu_num_data = pd.DataFrame(
            per_gpu_num_data, columns=VERSIONS_TO_KEEP)

        axes = per_gpu_num_data.plot.bar(
            colormap='Paired', color=colors, edgecolor='black', figsize=(15, 6))

        bars = axes.patches
        patterns = ('////', '\\\\\\', 'xxxx', '....')
        hatches = [p for p in patterns for i in range(len(per_gpu_num_data))]

        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        axes.set_ylabel('Time (s)')
        axes.set_title('Execution time per matrix')
        axes.legend()

        # axes.set(xlabel=None)
        axes.set(title=None)
        # wrap_labels(axes, 10)

        axes.legend(loc='upper center',
                    bbox_to_anchor=(0.5,  # horizontal
                                    1.09),  # vertical
                    ncol=2, fancybox=True)

        # per_gpu_num_title = f'{title} ({matrix_name})'
        per_gpu_num_title = gpu_num_column_label

        plt.xticks(rotation=-20, ha='center', weight='bold')
        # plt.suptitle(per_gpu_num_title)
        plt.savefig(MODULE_DIR / per_gpu_num_title)

    plt.show()
