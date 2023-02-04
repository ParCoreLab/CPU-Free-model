import argparse
import textwrap
from pathlib import Path

BASE_DIR = Path('Images')
BASE_DIR.mkdir(exist_ok=True)


def rotate(l, n):
    return l[-n:] + l[:-n]


def get_files():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=argparse.FileType('r'), nargs='+')
    return parser.parse_args().files


def get_module_dir(dir_name):
    module_dir = BASE_DIR / dir_name
    module_dir.mkdir(exist_ok=True)
    return module_dir


def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                                    break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)


markers = [
    '.',  # point
    ',',  # pixel
    'o',  # circle
    'v',  # triangle down
    '^',  # triangle up
    '<',  # triangle_left
    '>',  # triangle_right
    '1',  # tri_down
    '2',  # tri_up
    '3',  # tri_left
    '4',  # tri_right
    '8',  # octagon
    's',  # square
    'p',  # pentagon
    '*',  # star
    'h',  # hexagon1
    'H',  # hexagon2
    '+',  # plus
    'x',  # x
    'D',  # diamond
    'd',  # thin_diamond
    '|',  # vline
]
