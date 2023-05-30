import math
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import get_files, markers, get_module_dir, wrap_labels, rotate

from matplotlib.ticker import FormatStrFormatter

MODULE_DIR = get_module_dir('Weak Scaling')

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')

plt.rcParams.update({"axes.facecolor": (0.5, 0.5, 0.5, 0.1)})

plt.rcParams['text.usetex'] = True

MICROSECOND = 1000000

#NUM_ITERS = [1_000_000, 1_000_000, 10_000]
NUM_ITERS = [100_000, 100_000, 10_000]

files = get_files()

plots = len(files)
fig, axes = plt.subplots(math.ceil(plots / 3), plots if plots < 3 else 3, layout='constrained')
# fig.set_size_inches(15, 3 * math.ceil(plots / 3))
fig.set_size_inches(15, 3 * math.ceil(plots / 3))
# fig.tight_layout()

titles = ['Weak Scaling', 'Strong Scaling (No Compute) ($512^3$)', 'Strong Scaling ($256^3$)',]

logy = [False, True, True]

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = rotate(list(reversed(colors)), 1)

colors[1] = colors[-1]

ok = False

for ax, file, num_iter, title, logy in zip(axes.flatten(), files, NUM_ITERS, titles, logy):
#    ax.margins(x=0)

    data = pd.read_csv(file, index_col='Version')
#    data = data.sort_index()
    data = data.T / num_iter * MICROSECOND

    ax = data.plot(ax=ax, color=colors, title=title, logy=logy)

    if logy:
        ax.set_yscale('log', base=2)

    markers_cycle = cycle(markers)

    for line in ax.get_lines():
        line.set_marker(next(markers_cycle))
        line.set_linewidth(1.5)
        # If our versions
        if line.get_label().lower().startswith('baseline'):
            # line.set_linewidth(1.0)
            # line.set(alpha=0.5)
            line.set_linestyle('dashed')

#    axes.legend(axes.get_lines(), data.columns, loc='best')
    if ok:
        ax.set_xlabel('Number of GPUs', weight='bold', fontdict={'fontsize': 11.0})

    ok = True

    ax.get_legend().remove()
    wrap_labels(ax, 10)
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
