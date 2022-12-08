#!/bin/bash

#SBATCH --job-name=cg-bench
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --partition=dgx2q
#SBATCH --time=06:00:00
#SBATCH --output=sbatch_output_%j.log

NUM_ITER=${NUM_ITER:-1000}
NUM_RUNS=${NUM_RUNS:-5}
FILENAME=${FILENAME:-USE_DEFAULT_FILENAME}
MATRICES_FOLDER=${MATRICES_FOLDER:-USE_DEFAULT_MATRICES_FOLDER}
NUM_GPUS=${NUM_GPUS:-8}
GPU_MODEL=${GPU_MODEL:-V100}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi

  shift
done

cd ~/multi-perks/CG

. ./batch/_load_simula_modules.sh > /dev/null

echo "--- RUNNING ---"
date

python3 ./scripts/measure_runtime.py --num_iter $NUM_ITER --num_runs $NUM_RUNS --filename $FILENAME --matrices_folder $MATRICES_FOLDER --num_gpus $NUM_GPUS --gpu_model $GPU_MODEL

echo ""

echo "--- DONE ---"
date