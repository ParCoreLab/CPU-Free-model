import math
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import get_files, markers, get_module_dir, wrap_labels

MODULE_DIR = get_module_dir('Weak Scaling')

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')

files = get_files()

plots = len(files)
fig, axes = plt.subplots(math.ceil(plots / 3), 3)
fig.set_size_inches(18, 4 * math.ceil(plots / 3))
# fig.set_size_inches(plots * 5, 4)

for ax, file in zip(axes.flatten(), files):
    data = pd.read_csv(file, index_col='Version')
    # data = data.sort_index()
    data = data.T

    axes = data.plot(ax=ax)

    markers_cycle = cycle(markers)

    for line in axes.get_lines():
        line.set_marker(next(markers_cycle))
        # line.set_linewidth(2.5)
        # If our versions
        if line.get_label().lower().startswith('baseline'):
            # line.set(alpha=0.5)
            line.set_linestyle('dashed')

    # axes.legend(axes.get_lines(), data.columns, loc='best')
    axes.get_legend().remove()
    wrap_labels(axes, 10)

    # plt.xticks(fontsize=15)
    # plt.title(title, fontsize=15)

handles, labels = axes.get_legend_handles_labels()

# fig.legend(handles, labels, loc='upper center')

legend = fig.legend(handles, labels, loc='upper center',
                    bbox_to_anchor=(0.5,  # horizontal
                                    1.1),  # vertical
                    ncol=3, fancybox=True)

title = Path(files[0].name).stem

plt.tight_layout()
plt.savefig(MODULE_DIR / title, bbox_extra_artists=(legend,), bbox_inches='tight')

plt.show()
