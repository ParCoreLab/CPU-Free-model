#!/bin/bash

#SBATCH --job-name=stencil-bench
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --partition hgx2q
#SBATCH --time=01:00:00
#SBATCH --output=sbatch_output.log

. ./scripts/modules.sh

BIN="./jacobi -s 1"
V_SINGLE=6
V_OURS=1
V_BASELINE=6

NXNY="20480"

MAX_NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
CUDA_VISIBLE_DEVICES_SETTING=("0" "0,1" "0,1,2" "0,1,2,3" "0,1,2,3,4" "0,1,2,3,4,5" "0,1,2,3,4,5,6" "0,1,2,3,4,5,6,7" )
DOMAIN_SIZES=("256" "1024" "2048" "4096" "8192" "16384")
NUM_ITERS="100000"

function runp() {
    echo "$1"
    eval "$1"
}

for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS+=1 )); do
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}

    echo Baseline "$NUM_ITERS" iterations "$NUM_GPUS" GPUs
    # Baseline
    for d in "${DOMAIN_SIZES[@]}"; do
        runp "$BIN -v $V_BASELINE -nx $d -ny $d -niter $NUM_ITERS"
    done

    echo Ours "$NUM_ITERS" iterations "$NUM_GPUS" GPUs
    # Our version
    for d in "${DOMAIN_SIZES[@]}"; do
        runp "$BIN -v $V_OURS -nx $d -ny $d -niter $NUM_ITERS"
    done

    printf '\n'
done

