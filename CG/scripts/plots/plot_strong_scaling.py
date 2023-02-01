
import math
from itertools import cycle, islice
from os.path import dirname, realpath

from pathlib import Path

from common import get_files, markers, get_module_dir, wrap_labels

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

mpl.rcParams['path.simplify'] = True

MATRIX_NAMES = [
    # 'tridiagonal',
    # 'ecology2',
    #   'shallow_water2', Too little non-zeros
    #   'Trefethen_2000', Too little non-zeros
    # 'hood',
    # 'bmwcra_1',
    # 'consph',
    # 'thermomech_dM',
    # 'tmt_sym',
    # 'crankseg_1',
    # 'crankseg_2',
    'Queen_4147',
    # 'Bump_2911',
    # 'G3_circuit',
    # 'StocF-1465',
    # 'Flan_1565',
    # 'audikw_1',
    # 'Serena',
    # 'Geo_1438',
    # 'Hook_1498',
    #   'bone010', Multi-part matrix, don't handle those for now
    # 'ldoor'
]

LARGE_MATRICES_TO_PLOT = [
    'Queen_4147',
    'Bump_2911',
    'Flan_1565',
    'Hook_1498',
    'Flan_1565',
    'ldoor'
]

MEDIUM_SMALL_MATRICES_TO_PLOT = [
    'ecology2',
    'G3_circuit',
    'StocF-1465',
]

VERSIONS_TO_KEEP = [
    '(Baseline) Discrete Standard',
    '(Baseline) Discrete Pipelined',
    '(Ours) Persistent Standard',
    '(Ours) Persistent Pipelined'
]

MATRICES_TO_PLOT = LARGE_MATRICES_TO_PLOT + MEDIUM_SMALL_MATRICES_TO_PLOT

MATRICES_TO_PLOT = [
    'Queen_4147',
    'crankseg_2',
    'G3_circuit'
]

MODULE_DIR = get_module_dir('Strong Scaling')

dir_path = dirname(realpath(__file__))

plt.style.use(
    'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')
plt.rcParams['savefig.dpi'] = 300

plots = len(MATRICES_TO_PLOT)
fig, axes = plt.subplots(math.ceil(plots / 3), 3, layout='constrained')
fig.set_size_inches(18, 4 * math.ceil(plots / 3))

files = get_files()

for file in files:
    title = file.readline().strip()

    data = pd.read_csv(file, index_col='Version')
    data = data.sort_index()

    for ax, matrix_name in zip(axes.flatten(), MATRICES_TO_PLOT):
        per_matrix_data = data.loc[data['Matrix'] == matrix_name]
        per_matrix_data = per_matrix_data.drop(columns=['Matrix'])
        per_matrix_data = per_matrix_data.T
        per_matrix_data = pd.DataFrame(
            per_matrix_data, columns=VERSIONS_TO_KEEP)

        tmp_axes = per_matrix_data.plot(ax=ax, linewidth=2.0)
        tmp_axes.set_title(matrix_name, weight='bold')

        markers_cycle = cycle(markers)

        # Skip first few markers since they are underwhelming
        offset_markers_cycle = islice(markers_cycle, 10, None)

        for line in tmp_axes.get_lines():
            line.set_marker(next(offset_markers_cycle))

            # If baseline version
            if line.get_label().lower().startswith('(baseline)'):
                line.set_linestyle('dashed')

        tmp_axes.get_legend().remove()
        wrap_labels(tmp_axes, 10)

        # per_matrix_title = f'{title} ({matrix_name})'
        per_matrix_title = matrix_name

    handles, labels = tmp_axes.get_legend_handles_labels()

    handles, labels = ax.get_legend_handles_labels()
    axes.flatten()[2].legend(loc='best',
                             fancybox=True, prop={'weight': 'bold', 'size': 'large'})

    title = Path(files[0].name).stem

    y_label = fig.supylabel('Time (s)', weight='bold')

    plt.savefig(
        MODULE_DIR / (title + '.pdf'), format='pdf', dpi=600)

    plt.show()
