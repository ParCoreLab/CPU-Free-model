import sys
from os.path import dirname, realpath, basename

import scipy.io, PetscBinaryIO

if __name__ == "__main__":
    dir_path = dirname(realpath(__file__))

    mtx_matrix_path = None
    petsc_matrix_path = None

    arg_idx = 1

    while arg_idx < len(sys.argv):
        if sys.argv[arg_idx] == '--mtx_matrix_path':
            arg_idx += 1
            arg_val = sys.argv[arg_idx]

            mtx_matrix_path = arg_val

        if sys.argv[arg_idx] == '--petsc_matrix_path':
            arg_idx += 1
            arg_val = sys.argv[arg_idx]

            petsc_matrix_path = arg_val

        arg_idx += 1

    mtx_matrix = scipy.io.mmread(mtx_matrix_path)
    PetscBinaryIO.PetscBinaryIO().writeMatSciPy(open(petsc_matrix_path,'w'), mtx_matrix)