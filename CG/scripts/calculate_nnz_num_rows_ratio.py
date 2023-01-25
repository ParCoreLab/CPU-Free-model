# Maps matrix name to tuple (num_rows, num_nnz)
MATRIX_NAMES_TO_INFO = {
    'ecology2': (999999, 4995991),
    'hood': (220542, 10768436),
    'bmwcra_1': (148770, 10644002),
    'consph': (83334, 6010480),
    'thermomech_dM': (204316, 1423116),
    'tmt_sym': (726713, 5080961),
    'crankseg_1': (52804, 10614210),
    'crankseg_2': (63838, 14148858),
    'Queen_4147': (4147110, 329499284),
    'Bump_2911': (2911419, 127729899),
    'G3_circuit': (1585478, 7660826),
    'StocF-1465': (1465137, 21005389),
    'Flan_1565': (1564794, 117406044),
    'audikw_1': (943695, 77651847),
    'Serena': (1391349, 64531701),
    'Geo_1438': (1437960, 63156690),
    'Hook_1498': (1498023, 60917445),
    'ldoor': (952203, 46522475)
}

if __name__ == "__main__":
    for matrix_name, (num_rows, num_nnz) in MATRIX_NAMES_TO_INFO.items():
        nnz_to_num_rows_ratio = num_nnz / num_rows
        print(
            f'Sparsity for matrix {matrix_name} is {nnz_to_num_rows_ratio:.2f}')
