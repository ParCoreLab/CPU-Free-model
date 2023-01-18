from itertools import cycle

import matplotlib.pyplot as plt
import pandas as pd

from common import get_files, markers, get_module_dir

MODULE_DIR = get_module_dir('Weak Scaling')

plt.style.use('./paper.mplstyle')

files = get_files()

for file in files:
    title = file.readline().strip()

    data = pd.read_csv(file, index_col='Version')
    data = data.sort_index()
    data = data.T

    axes = data.plot()

    markers_cycle = cycle(markers)

    for line in axes.get_lines():
        line.set_marker(next(markers_cycle))

        # If our versions
        if line.get_label().lower().startswith('baseline'):
            line.set_linestyle('dashed')

    axes.legend(axes.get_lines(), data.columns, loc='best')

    plt.title(title)
    plt.savefig(MODULE_DIR / title)

    plt.show()
