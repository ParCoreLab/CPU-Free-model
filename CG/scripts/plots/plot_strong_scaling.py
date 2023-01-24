
import math
from itertools import cycle
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

        axes = per_matrix_data.plot(ax=ax)

        markers_cycle = cycle(markers)

        for line in axes.get_lines():
            line.set_marker(next(markers_cycle))

            # If baseline version
            if line.get_label().lower().startswith('baseline'):
                line.set_linestyle('dashed')

            markers_cycle = cycle(markers)

        axes.get_legend().remove()
        wrap_labels(axes, 10)

        # per_matrix_title = f'{title} ({matrix_name})'
        per_matrix_title = matrix_name

    handles, labels = axes.get_legend_handles_labels()

    legend = fig.legend()

    title = Path(files[0].name).stem

    plt.tight_layout()
    plt.savefig(MODULE_DIR / title, bbox_extra_artists=(legend,),
                bbox_inches='tight')

    plt.show()
