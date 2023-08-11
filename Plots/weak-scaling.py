import math
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import get_files, markers, get_module_dir, wrap_labels, rotate

MODULE_DIR = get_module_dir('Weak Scaling')

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')

plt.rcParams.update({"axes.facecolor": (0.5, 0.5, 0.5, 0.1)})

plt.rcParams['text.usetex'] = True


MICROSECOND = 1000000

#NUM_ITERS = [1_000_000, 1_000_000, 10_000]
NUM_ITERS = [1_000_000, 1_000_000, 1_000_000, 10_000]

files = get_files()

plots = len(files)
fig, axes = plt.subplots(math.ceil(plots / 3), plots if plots < 3 else 3, layout='constrained')
# fig.set_size_inches(15, 3 * math.ceil(plots / 3))
fig.set_size_inches(13, 3 * math.ceil(plots / 3))
# fig.tight_layout()

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = rotate(list(reversed(colors)), 1)

colors[1] = colors[-1]

for ax, file, num_iter in zip(axes.flatten(), files, NUM_ITERS):
    data = pd.read_csv(file, index_col='Version')
#    data = data.sort_index()
    data = data.T / num_iter * MICROSECOND

    ax = data.plot(ax=ax, color=colors)

    markers_cycle = cycle(markers)

    for line in ax.get_lines():
        line.set_marker(next(markers_cycle))
        line.set_linewidth(1.5)
        # If our versions
        if line.get_label().lower().startswith('baseline'):
#            line.set_linewidth(1.0)
            # line.set(alpha=0.5)
            line.set_linestyle('dashed')

    # axes.legend(axes.get_lines(), data.columns, loc='best')
    ax.get_legend().remove()
    wrap_labels(ax, 10)

    # plt.xticks(fontsize=15)
    # plt.title(title, fontsize=15)

# handles, labels = axes.get_legend_handles_labels()

axes.flatten()[0].legend(loc='best', fancybox=True, prop={'weight': 'bold', 'size': 'large'})
axes.flatten()[0].legend(loc='best', fancybox=True)

# legend = fig.legend(handles, labels, loc='upper center',
#                     bbox_to_anchor=(0.5,  # horizontal
#                                     1.1),  # vertical
#                     ncol=6, fancybox=True)

#legend = fig.legend(handles, labels, loc='best')

fig.supylabel(r'$\mu$ seconds per iteration', weight='normal')

title = Path(files[0].name).stem

format = 'pdf'
#plt.constrained_layoadia
plt.savefig(MODULE_DIR / f'{title}.{format}', bbox_inches='tight', format=format, transparent=False)

plt.show()
