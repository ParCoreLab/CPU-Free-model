import argparse
import textwrap
from pathlib import Path
from os.path import dirname, realpath

dir_path = dirname(realpath(__file__))

BASE_DIR = Path(dir_path + '/../../img')
BASE_DIR.mkdir(exist_ok=True)


def get_files():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=argparse.FileType('r'), nargs='+')
    return parser.parse_args().files


def get_module_dir(dir_name):
    module_dir = BASE_DIR / dir_name
    module_dir.mkdir(exist_ok=True)
    return module_dir


def wrap_labels(ax, width, break_long_words=False):
    x_labels = []
    y_labels = []

    for x_label in ax.get_xticklabels():
        text = x_label.get_text()
        x_labels.append(textwrap.fill(text, width=width,
                                      break_long_words=break_long_words))

    ax.set_xticklabels(x_labels, rotation=0)

    for y_label in ax.get_yticklabels():
        text = y_label.get_text()
        y_labels.append(textwrap.fill(text, width=width,
                                      break_long_words=break_long_words))
    ax.set_yticklabels(y_labels, rotation=0)


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
