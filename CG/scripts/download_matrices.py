from io import BytesIO
from urllib.request import urlopen
import tarfile

from os.path import dirname, realpath, basename
import os
import sys

SAVE_MATRICES_TO_FOLDER = None

SUITE_SPARSE_BASE_URL = 'https://suitesparse-collection-website.herokuapp.com/MM'

MATRIX_INDICES = [
    'McRae/ecology2',
    'GHS_psdef/hood',
    'GHS_psdef/bmwcra_1',
    'Williams/consph',
    'Botonakis/thermomech_dM',
    'CEMW/tmt_sym',
    'GHS_psdef/crankseg_1',
    'GHS_psdef/crankseg_2',
    'TKK/cbuckle',
    'BenElechi/BenElechi1',
    'MaxPlanck/shallow_water2',
    'JGD_Trefethen/Trefethen_2000',
    'Janna/Queen_4147',
    'Janna/Bump_2911',
    'AMD/G3_circuit',
    'Janna/StocF-1465',
    'Janna/Flan_1565',
    'GHS_psdef/audikw_1',
    'Janna/Serena',
    'Janna/Geo_1438',
    'Janna/Hook_1498',
    'Oberwolfach/bone010',
    'GHS_psdef/ldoor',
]


def download_matrices():
    for matrix_index in MATRIX_INDICES:
        matrix_name = matrix_index.split('/')[-1]

        mtx_filename = f'{matrix_name}.mtx'
        mtx_filepath = f'{SAVE_MATRICES_TO_FOLDER}/{mtx_filename}'

        if os.path.exists(mtx_filepath):
            print(f'Matrix {matrix_name} is already downloaded')
            continue

        # Check here is matrix is already downloaded
        matrix_url = f'{SUITE_SPARSE_BASE_URL}/{matrix_index}.tar.gz'

        with urlopen(matrix_url) as zip_response:
            zip_file = tarfile.open(fileobj=zip_response, mode='r|gz')

            zip_file.extractall(SAVE_MATRICES_TO_FOLDER)

            # It's possible that matrices are extracted as
            # 1) matix_name/matrix_name.mtx
            # 2) or as matrix_name.mtx
            # In the second case, we don't need to create a temp folder
            if not os.path.exists(mtx_filepath):
                for zip_file_member in zip_file.getmembers():
                    member_name = zip_file_member.name

                    old_path = f'{SAVE_MATRICES_TO_FOLDER}/{member_name}'

                    tmp_folder_path = f'{SAVE_MATRICES_TO_FOLDER}/{os.path.dirname(member_name)}'

                    os.rename(old_path, mtx_filepath)

                os.rmdir(tmp_folder_path)

            print(f'Downloaded matrix {matrix_name}')


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
