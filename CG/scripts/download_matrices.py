from io import BytesIO
from urllib.request import urlopen
import tarfile

from os.path import dirname, realpath, basename
import os
import sys

SAVE_MATRICES_TO_FOLDER = None

SUITE_SPARSE_BASE_URL = 'https://suitesparse-collection-website.herokuapp.com/MM'

MATRIX_NAME_TO_INDEX_MAP = {
    'ecology2': 'McRae/ecology2',
    'hood': 'GHS_psdef/hood',
    'bmwcra_1': 'GHS_psdef/bmwcra_1',
    'consph': 'Williams/consph',
    'thermotech_dM': 'Botonakis/thermomech_dM',
    'tmt_sym': 'CEMW/tmt_sym',
    'crankseg_1': 'GHS_psdef/crankseg_1',
    'crankseg_2': 'GHS_psdef/crankseg_2',
    'cbuckle': 'TKK/cbuckle',
    'BenElechi1': 'BenElechi/BenElechi1',
    'shallow_water': 'MaxPlanck/shallow_water2',
    'Trefethen_2000': 'JGD_Trefethen/Trefethen_2000',
}


def download_matrices():
    for matrix_name, matrix_index in MATRIX_NAME_TO_INDEX_MAP.items():
        matrix_url = f'{SUITE_SPARSE_BASE_URL}/{matrix_index}.tar.gz'

        with urlopen(matrix_url) as zip_response:
            zip_file = tarfile.open(fileobj=zip_response, mode='r|gz')

            zip_file.extractall(SAVE_MATRICES_TO_FOLDER)

            for zip_file_member in zip_file.getmembers():
                member_name = zip_file_member.name

                old_path = f'{SAVE_MATRICES_TO_FOLDER}/{member_name}'

                tmp_folder_path = f'{SAVE_MATRICES_TO_FOLDER}/{os.path.dirname(member_name)}'
                mtx_basename = basename(member_name)

                new_path = f'{SAVE_MATRICES_TO_FOLDER}/{mtx_basename}'

                os.rename(old_path, new_path)

            os.rmdir(tmp_folder_path)


if __name__ == "__main__":
    dir_path = dirname(realpath(__file__))

    arg_idx = 1

    while arg_idx < len(sys.argv):
        if sys.argv[arg_idx] == '--save_matrices_to_folder':
            arg_idx += 1
            arg_val = sys.argv[arg_idx]

            SAVE_MATRICES_TO_FOLDER = arg_val

        arg_idx += 1

    download_matrices()
