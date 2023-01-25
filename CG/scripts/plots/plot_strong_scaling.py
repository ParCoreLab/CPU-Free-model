
import math
from itertools import cycle, islice
from os.path import dirname, realpath

from pathlib import Path

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
    # 'thermomech_dM',
    # 'tmt_sym',
    # 'crankseg_1',
    # 'crankseg_2',
    # 'Queen_4147',
    # 'Bump_2911',
    # 'G3_circuit',
    # 'StocF-1465',
    # 'Flan_1565',
    # 'audikw_1',
    # 'Serena',
    # 'Geo_1438',
    # 'Hook_1498',
    #   'bone010', Multi-part matrix, don't handle those for now
    'ldoor'
]

VERSIONS_TO_KEEP = [
    '(Baseline) Discrete Standard',
    '(Baseline) Discrete Pipelined',
    '(Ours) Persistent Standard',
    '(Ours) Persistent Pipelined'
]

MODULE_DIR = get_module_dir('Strong Scaling')

dir_path = dirname(realpath(__file__))

plt.style.use(
    'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')

plots = len(MATRIX_NAMES)
fig, axes = plt.subplots(math.ceil(plots / 3), 3)
fig.set_size_inches(18, 4 * math.ceil(plots / 3))

files = get_files()

for file in files:
    title = file.readline().strip()

    data = pd.read_csv(file, index_col='Version')
    data = data.sort_index()

    for ax, matrix_name in zip(axes.flatten(), MATRIX_NAMES):
        per_matrix_data = data.loc[data['Matrix'] == matrix_name]
        per_matrix_data = per_matrix_data.drop(columns=['Matrix'])
        per_matrix_data = per_matrix_data.T
        per_matrix_data = pd.DataFrame(
            per_matrix_data, columns=VERSIONS_TO_KEEP)

        tmp_axes = per_matrix_data.plot(ax=ax)
        tmp_axes.set_title(matrix_name)

        markers_cycle = cycle(markers)

        # Skip first few markers since they are underwhelming
        offset_markers_cycle = islice(markers_cycle, 11, None)

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

    legend = fig.legend(handles, labels, loc='upper center',
                        ncol=2)

    title = Path(files[0].name).stem

    plt.tight_layout()
    plt.savefig(MODULE_DIR / title, bbox_extra_artists=(legend,),
                bbox_inches='tight')

    plt.show()
