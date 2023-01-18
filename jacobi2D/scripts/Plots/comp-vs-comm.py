import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import cycle

from common import get_module_dir, wrap_labels

MODULE_DIR = get_module_dir('Comp vs Comm')

plt.style.use('./paper.mplstyle')

data_comp = pd.read_csv('data/comp.csv', index_col='Version')
data_no_comp = pd.read_csv('data/no-comp.csv', index_col='Version')

# Make sure both have the same version names
data_no_comp.index = data_comp.index.copy()


def plot_one_gpu(comp, no_comp, title):
    # Normalize to 100%
    # Percentage
    no_comp_left = (no_comp / comp) * 100
    # comp_left = 100 - no_comp_left
    comp_left = 100 - no_comp_left

    # no_comp_left += comp_left

    # Actual execution time
    comp_right = comp #- no_comp

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(10, 6)

    data_left = pd.DataFrame({'Comp %': comp_left, 'Comm %': no_comp_left, 'idx_col': comp_left.index})
    data_right = pd.DataFrame(
        {'Comp sec.': comp_right, 'Comm sec.': no_comp, 'idx_col': comp_right.index}
    )

    indices = np.arange(3)

    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'][:2])
    widths = cycle([0.8, 0.6])

    # data_left.iloc[:, 1].plot.bar(indices, color='r', width=0.8, stacked=True)
    # comp_left.plot.bar(indices, color='b', width=0.6, stacked=True)
    # plt.show()

    for ax, data in zip(axes, [data_left, data_right]):

        for i in range(2):
            data.iloc[:, i].plot.bar(ax=ax, color=next(colors), width=next(widths))

        # data.plot.bar(ax=ax, width=0.6, stacked=True)

        # Hatch stuff
        bars = ax.patches

        bars[0].set_width(0.8)
        bars[1].set_width(0.8)

        hatches = ''.join(h * len(data_left) for h in [' ', '/'])

        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        # Beautify stuff
        ax.set(xlabel=None)
        wrap_labels(ax, 10)

        ax.legend(loc='upper center',
                  bbox_to_anchor=(0.5,  # horizontal
                                  1.09),  # vertical
                  ncol=3, fancybox=True)

    plt.xticks(rotation=0, ha='center')
    fig.suptitle(title)
    plt.savefig(MODULE_DIR / title)
    plt.show()


for (title, comp), (_, no_comp) in zip(data_comp.iteritems(), data_no_comp.iteritems()):
    plot_one_gpu(comp=comp, no_comp=no_comp, title=title)
