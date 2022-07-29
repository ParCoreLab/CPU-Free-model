import re
import sys
from collections import defaultdict

import pandas as pd

DELIMITER = '-------------------------------------'

results_path = sys.argv[1]


def print_data_tabular(version_to_result_map, column_labels):
    row_labels = version_to_result_map.keys()
    full_perf_data = version_to_result_map.values()
    transposed_perf_data = list(zip(*full_perf_data))

    df = pd.DataFrame(full_perf_data, columns=column_labels,
                      index=row_labels)

    df.to_csv(sys.stdout)


with open(results_path) as file:
    results = file.read()

results_per_version = results.split(DELIMITER)[:-1]
results_per_version = [result.strip() for result in results_per_version]

version_to_result_map = defaultdict(list)
num_gpus_grid_size_label = []

for version_result in results_per_version:
    chunks = version_result.split('\n\n')
    version_name = ' '.join(chunks[0].split()[1:])
    x_axis_label = []

    for data_chunk in chunks[1:]:
        chunk_lines = data_chunk.splitlines()
        num_gpus = int(re.match("Num GPUS: (?P<num_gpus>\d+)",
                                chunk_lines[0]).group('num_gpus'))

        run_parameters_match = re.match(
            "(?P<num_iter>\d+) iterations on grid (?P<nx>\d+)x(?P<ny>\d+)", chunk_lines[1])

        num_iterations = int(run_parameters_match.group('num_iter'))
        grid_nx = int(run_parameters_match.group('nx'))
        grid_ny = int(run_parameters_match.group('ny'))

        if not num_gpus_grid_size_label:
            label = f"{num_gpus} GPU" + \
                ("s" if num_gpus > 1 else "") + f" ({grid_nx}x{grid_ny})"
            x_axis_label.append(label)

        perf_data_pattern = re.compile(
            "Execution time:\s+(?P<exec_time>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?) s on run (?P<run_num>\d+)")

        execution_times = []
        run_idx = 2

        while run_idx < len(chunk_lines) and (perf_data_match := perf_data_pattern.match(chunk_lines[run_idx])):
            exec_time = float(perf_data_match.group('exec_time'))
            execution_times.append(exec_time)

            run_idx += 1

        min_execution_time = min(execution_times)
        version_to_result_map[version_name].append(min_execution_time)

    if not num_gpus_grid_size_label:
        num_gpus_grid_size_label[:] = x_axis_label

print_data_tabular(version_to_result_map, num_gpus_grid_size_label)
