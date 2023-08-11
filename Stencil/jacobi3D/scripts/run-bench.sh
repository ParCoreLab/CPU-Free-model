#!/bin/bash

#SBATCH --job-name=stencil-bench
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --partition hgx2q
#SBATCH --time=01:30:00
#SBATCH --output=sbatch_output_%j.log

SCRIPT="./scripts/weak_scale_bench.sh"

ARGS=(
    "256 256 128 10000"
    "256 256 256 10000"
    "512 512 32 1000"
    "512 512 512 100"
)

for i in "${ARGS[@]}"
do
    "$SCRIPT" $i;
    printf '\n\n'
done
