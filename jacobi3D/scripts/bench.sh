#!/bin/bash

#SBATCH --job-name=stencil-bench
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --partition hgx2q
#SBATCH --time=01:00:00
#SBATCH --output=sbatch_output_%j.log

. ./scripts/modules.sh > /dev/null

BIN="./jacobi -s 1"
NUM_RUNS=5
V_OURS=1
V_BASELINE=6

#OUT_CSV="./results.csv"
OUT_CSV="/dev/stdout"
echo "version,nx,ny,niter,num_gpus,execution_time" >> "$OUT_CSV"

#MAX_NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
MAX_NUM_GPUS=4

# First element reserved for pretty output in the loop
CUDA_VISIBLE_DEVICES_SETTING=("x" "0" "0,1" "0,1,2,3" "0,1,2,3,4,5,6,7")
DOMAIN_SIZES=(
    "x"
    "8192 4096"
    "8192 8192"
    "8192 16348"
    "16348 16348"
)
NUM_ITERS=100000

function runp() {
    cmd="$BIN -v $1 -nx $2 -ny $3 -niter $4"

    min_execution_time=9223372036854775807

    for ((i = 0; i < NUM_RUNS; i += 1)); do
        execution_time=$($cmd | grep -o -E "[0-9]+.?[0-9]+")
        min_execution_time=$(python -c "print(min($execution_time, $min_execution_time))")
    done

    echo "$1,$2,$3,$4,$5,$min_execution_time" >> "$OUT_CSV"
}

for ((NUM_GPUS = 1; NUM_GPUS <= MAX_NUM_GPUS; NUM_GPUS += 1)); do
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}

    read -r nx ny <<< "${DOMAIN_SIZES[$NUM_GPUS]}"

    # Our version
    runp "$V_OURS" "$nx" "$ny" "$NUM_ITERS" "$NUM_GPUS"

    # Baseline
    runp "$V_BASELINE" "$nx" "$ny" "$NUM_ITERS" "$NUM_GPUS"
done
