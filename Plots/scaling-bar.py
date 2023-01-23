from itertools import cycle

import matplotlib.pyplot as plt
import pandas as pd

import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.3

import common
from common import get_files, markers, get_module_dir, wrap_labels

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')

MODULE_DIR = get_module_dir('Bar Scaling')

files = get_files()

for file in files:
    data = pd.read_csv(file, index_col='Version')
    data = data.sort_index()

    data = data.T

    colors = [
        '#c6c9cb', '#64b8e5', '#ee7fb2'
    ]

    axes = data.plot.bar(colormap='Paired', color=colors, edgecolor='black')

    bars = axes.patches
    patterns = ('///', '\\\\\\', 'xxx')
    hatches = [p for p in patterns for i in range(len(data))]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # for line in axes.get_lines():
    #     line.set_hatch(next(markers_cycle))
    #     line.set_linewidth(2)
    #     line.set(alpha=0.5)
    #     line.set(color=next(colors))
    #
    #     # If our versions
    #     if line.get_label().lower().startswith('baseline'):
    #         # line.set(alpha=0.5)
    #         line.set_linestyle('dashed')
    #
    # # axes.legend(axes.get_lines(), data.columns, loc='best')
    # wrap_labels(axes, 10)
    #
    # # plt.xticks(fontsize=15)
    # plt.title(title)
    # plt.savefig(MODULE_DIR / title)

    # plt.grid(axis='x')

    plt.show()
