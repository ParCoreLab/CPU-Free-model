
from itertools import cycle
from os.path import dirname, realpath

from common import get_files, markers, get_module_dir, wrap_labels

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

MODULE_DIR = get_module_dir('Operation Breakdown')

dir_path = dirname(realpath(__file__))

# plt.style.use(dir_path + '/default.mplstyle')
plt.style.use('fivethirtyeight')

files = get_files()

for file in files:
    title = file.readline().strip()

    operation_breakdowns = pd.read_csv(file, index_col='Matrix')
    operation_breakdowns = operation_breakdowns.T
    operation_breakdowns = operation_breakdowns[MATRIX_NAMES]
    operation_breakdowns = operation_breakdowns.T

    # Is this necessary?
    # operation_breakdowns = operation_breakdowns.sort_index()

    # Get percentages of operations
    per_operation_percentages = operation_breakdowns.div(
        operation_breakdowns.sum(axis=1), axis=0) * 100

    ax = per_operation_percentages.plot.barh(stacked=True)
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(per_operation_percentages, axis=1).max())

    for container in ax.containers:
        ax.bar_label(container)

    wrap_labels(ax, 10, break_long_words=True)

    plt.title(title)
    plt.savefig(MODULE_DIR / title)

    plt.show()
