
import math
from itertools import cycle, islice
from os.path import dirname, realpath

from pathlib import Path

from common import get_files, markers, get_module_dir, wrap_labels, rotate

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

mpl.rcParams['path.simplify'] = True

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

LARGE_MATRICES_TO_PLOT = [
    'Queen_4147',
    'Bump_2911',
    'Flan_1565',
    'Hook_1498',
    'Flan_1565',
    'ldoor'
]

MEDIUM_SMALL_MATRICES_TO_PLOT = [
    'StocF-1465',
    'thermomech_dM',
    'hood',
    'ecology2',
    'G3_circuit',
    'tmt_sym',
]

VERSIONS_TO_KEEP = [
    'PETSc Standard CG (Baseline)',
    'PETSc Pipelined CG (Baseline)',
    'CPU-Free Standard CG (Ours)',
    'CPU-Free Pipelined CG (Ours)'
]

MATRICES_TO_PLOT = LARGE_MATRICES_TO_PLOT + MEDIUM_SMALL_MATRICES_TO_PLOT

# MATRICES_TO_PLOT = [
#     'Queen_4147',
#     'crankseg_2',
#     'G3_circuit'
# ]

MATRICES_TO_PLOT = [
    'Queen_4147',
    'crankseg_2',
    'G3_circuit',
]

MODULE_DIR = get_module_dir('PETSc')

dir_path = dirname(realpath(__file__))

plt.style.use(
    'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')
plt.rcParams.update({
    "axes.facecolor":    (0.5, 0.5, 0.5, 0.1),
})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = rotate(colors, 1)

plots = len(LARGE_MATRICES_TO_PLOT)
fig, axes = plt.subplots(math.ceil(plots / 3), 3, layout='constrained')
fig.set_size_inches(18, 3 * math.ceil(plots / 3))

files = get_files()

full_runtimes_csv = files[0]
single_gpu_runtimes_csv = files[1]

title = full_runtimes_csv.readline().strip()

data = pd.read_csv(full_runtimes_csv, index_col='Version')
data = data.sort_index()

single_gpu_baseline_standard_runtimes = pd.read_csv(
    single_gpu_runtimes_csv, index_col='Matrix')['Runtime']
single_gpu_baseline_standard_runtimes = single_gpu_baseline_standard_runtimes.sort_index()

for ax, matrix_name in zip(axes.flatten(), LARGE_MATRICES_TO_PLOT):
    per_matrix_data = data.loc[data['Matrix'] == matrix_name]
    per_matrix_data = per_matrix_data.drop(columns=['Matrix'])
    per_matrix_data = per_matrix_data.T
    per_matrix_data = pd.DataFrame(
        per_matrix_data, columns=VERSIONS_TO_KEEP)

    # per_matrix_data['Single GPU Standard CG'] = single_gpu_baseline_standard_runtimes[matrix_name]

    tmp_axes = per_matrix_data.plot(
        ax=ax, linewidth=1.5, logy=False, color=colors)
    tmp_axes.set_title(matrix_name, weight='bold', fontdict={'fontsize': 15.0})

    markers_cycle = cycle(markers)

    # Skip first few markers since they are underwhelming
    offset_markers_cycle = islice(markers_cycle, 0, None)

    for line in tmp_axes.get_lines():
        line.set_marker(next(offset_markers_cycle))

        # If baseline version
        if '(baseline)' in line.get_label().lower():
            line.set_linestyle('dashed')

        # if 'single gpu' in line.get_label().lower():
        #     line.set_linestyle('dashdot')
        #     line.set_color('black')
        #     line.set_marker('x')
        #     line.set_linewidth(1.5)

    tmp_axes.get_legend().remove()
    wrap_labels(tmp_axes, 10)

    # per_matrix_title = f'{title} ({matrix_name})'
    per_matrix_title = matrix_name

handles, labels = tmp_axes.get_legend_handles_labels()

handles, labels = ax.get_legend_handles_labels()
axes.flatten()[1].legend(loc='upper center',
                             fancybox=True, prop={'weight': 'bold', 'size': 'large'})

title = Path(files[0].name).stem

y_label = fig.supylabel('Time (s)', weight='bold')

plt.savefig(
    MODULE_DIR / (title + '.pdf'), format='pdf')

plt.show()
