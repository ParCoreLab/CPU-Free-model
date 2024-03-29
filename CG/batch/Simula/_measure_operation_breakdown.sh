#!/bin/bash

#SBATCH --job-name=cg-bench
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --partition=dgx2q
#SBATCH --time=06:00:00
#SBATCH --output=sbatch_output_%j.log

NUM_ITER=${NUM_ITER:-1000}
FILENAME=${FILENAME:-USE_DEFAULT_FILENAME}
MATRICES_FOLDER=${MATRICES_FOLDER:-USE_DEFAULT_MATRICES_FOLDER}
GPU_MODEL=${GPU_MODEL:-V100}

# This will be a comma delimited list of number of GPUs to run on
# No spaces between numbers
# Single numbers also work
# (Example => 2,3,4,8)
NUM_GPUS=${NUM_GPUS:-8}

# This will be a comma delimited list of version indices
# No spaces between numbers
# Single numbers also work
# (Example => 0,1,2,4)
# Runs all versions by default
VERSIONS_TO_RUN=${VERSIONS_TO_RUN:-RUN_ALL_VERSIONS}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi

  shift
done

cd ~/multi-perks/CG

. ./batch/Simula/_load_simula_modules.sh > /dev/null

echo "--- RUNNING ---"
date

python3 ./scripts/measure_operation_breakdown.py --num_iter $NUM_ITER --filename $FILENAME --matrices_folder $MATRICES_FOLDER --num_gpus $NUM_GPUS --versions_to_run $VERSIONS_TO_RUN --gpu_model $GPU_MODEL
rm ./nsys_reports/*

echo ""

echo "--- DONE ---"
date