
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from os.path import dirname, realpath

from common import get_files, markers, get_module_dir, wrap_labels, set_size, ACM_DOCUMENT_WIDTH, rotate

import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.3
mpl.rcParams['path.simplify'] = True
mpl.rcParams['figure.dpi'] = 300

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
    'CPU-Controlled Standard CG (Baseline)',
    'CPU-Controlled Pipelined CG (Baseline)',
    'CPU-Free Standard CG (Ours)',
    'CPU-Free Pipelined CG (Ours)'
]

MODULE_DIR = get_module_dir('Constant Number of GPUs')

dir_path = dirname(realpath(__file__))

plt.style.use(
    'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams.update({
    "axes.facecolor":    (0.5, 0.5, 0.5, 0.1),
})
# colors = [
#     '#c6c9cb', '#64b8e5', '#ee7fb2', '#eae2b7'
# ]

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = rotate(colors, 1)

# Color candidates =>
# - '#30becd' => Maximum Blue Green
# - '#eae2b7' => Yellow


# First file should be the full CSV file
# Second should be the SingleGPU runtimes
files = get_files()

full_runtimes_csv = files[0]
single_gpu_runtimes_csv = files[1]

title = full_runtimes_csv.readline().strip()

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
        single_gpu_baseline_standard_runtimes.drop(matrix_name, inplace=True)


for gpu_num_column_label in gpu_num_column_labels:
    per_gpu_num_data = data[['Version', gpu_num_column_label]]
    per_gpu_num_data = per_gpu_num_data.pivot_table(
        gpu_num_column_label, 'Matrix', 'Version')

    per_gpu_num_data = pd.DataFrame(
        per_gpu_num_data, columns=VERSIONS_TO_KEEP)

    per_gpu_num_speedup = 1 / per_gpu_num_data.div(
        single_gpu_baseline_standard_runtimes, axis=0)

    pipelined_cg_speedup = per_gpu_num_speedup['CPU-Free Pipelined CG (Ours)'] / \
        per_gpu_num_speedup['CPU-Controlled Pipelined CG (Baseline)']
    standard_cg_speedup = per_gpu_num_speedup['CPU-Free Standard CG (Ours)'] / \
        per_gpu_num_speedup['CPU-Controlled Standard CG (Baseline)']

    pipelined_cg_geo_mean_spedup = np.exp(np.log(pipelined_cg_speedup).mean())
    standard_cg_geo_mean_spedup = np.exp(np.log(standard_cg_speedup).mean())

    print(
        f'Pipelined CG geo mean speedup  for {gpu_num_column_label}: {pipelined_cg_geo_mean_spedup}')
    print(
        f'Standard CG geo mean speedup  for {gpu_num_column_label}: {standard_cg_geo_mean_spedup}')
    per_gpu_num_speedup.sort_values(
        inplace=True, by='CPU-Free Pipelined CG (Ours)', ascending=False)

    axes = per_gpu_num_speedup.plot.bar(
        color=colors, edgecolor='black', figsize=(12, 3))

    bars = axes.patches
    patterns = ('', '///', '', '///')
    hatches = [p for p in patterns for i in range(
        len(per_gpu_num_speedup))]

    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    axes.set_ylabel('Speedup', weight='bold')
    axes.set_xlabel(None)
    # axes.set_title(f'Speedup per matrix on {gpu_num_column_label}')
    axes.legend(fancybox=True, prop={'weight': 'bold', 'size': 'large'})

    # axes.set(xlabel=None)
    axes.set(title=None)
    # wrap_labels(axes, 10)

    # axes.legend(loc='upper center',
    #             bbox_to_anchor=(0.5,  # horizontal
    #                             1.09),  # vertical
    #             ncol=2, fancybox=True)

    # per_gpu_num_title = f'{title} ({matrix_name})'
    per_gpu_num_title = gpu_num_column_label

    plt.xticks(rotation=-20, ha='center', weight='bold')
    # plt.suptitle(per_gpu_num_title)
    plt.savefig(
        MODULE_DIR / ('matrix_speedup_table_' + per_gpu_num_title + '.pdf'), format='pdf')

plt.show()
