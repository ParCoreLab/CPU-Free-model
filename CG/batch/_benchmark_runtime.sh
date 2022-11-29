#!/bin/bash

#SBATCH --job-name=cg-bench
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --partition=dgx2q
#SBATCH --time=12:00:00
#SBATCH --output=sbatch_output_%j.log

NUM_ITER=${NUM_ITER:-100}
FILENAME=${FILENAME:-USE_DEFAULT_FILENAME}
MATRICES_FOLDER=${MATRICES_FOLDER:-USE_DEFAULT_MATRICES_FOLDER}
NUM_GPUS=${NUM_GPUS:-8}

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

./scripts/venv/bin/python3 ./scripts/benchmark_runtime.py --num_iter $NUM_ITER --filename $FILENAME --matrices_folder $MATRICES_FOLDER --num_gpus $NUM_GPUS

echo ""

echo "--- DONE ---"
date