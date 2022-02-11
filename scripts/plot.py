import sys
import re

DELIMITER = '-------------------------------------'

results_path = sys.argv[1]

with open(results_path) as file:
    results = file.read()

results_per_version = results.split(DELIMITER)[:-1]
results_per_version = [result.strip() for result in results_per_version]

version_to_result_map = {}

for version_result in results_per_version:
    chunks = version_result.split('\n\n')
    version_name = ' '.join(chunks[0].split()[1:])

    for data_chunk in chunks[1:]:
        chunk_lines = data_chunk.splitlines()
        num_gpus = int(re.match("Num GPUS: (?P<num_gpus>\d+)",
                       chunk_lines[0]).group('num_gpus'))

        run_parameters_match = re.match(
            "(?P<num_iter>\d+) iterations on grid (?P<nx>\d+)x(?P<ny>\d+)", chunk_lines[1])

        num_iterations = int(run_parameters_match.group('num_iter'))
        grid_nx = int(run_parameters_match.group('nx'))
        grid_ny = int(run_parameters_match.group('ny'))

        perf_data_pattern = re.compile(
            "Execution time:\s+(?P<exec_time>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?) s on run (?P<run_num>\d+)")

        execution_times = []
        run_idx = 2

        while run_idx < len(chunk_lines) and (perf_data_match := perf_data_pattern.match(chunk_lines[run_idx])):
            exec_time = float(perf_data_match.group('exec_time'))
            execution_times.append(exec_time)

            run_idx += 1

        min_execution_time = min(execution_times)

        version_to_result_map
