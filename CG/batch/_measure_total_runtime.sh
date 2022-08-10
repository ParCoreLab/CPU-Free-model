#!/bin/bash

#SBATCH --job-name=cg-bench
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --partition=dgx2q
#SBATCH --time=06:00:00
#SBATCH --output=sbatch_output_%j.log

NUM_ITER=${NUM_ITER:-1000000}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi

  shift
done

cd ~/multi-perks/CG

. ./scripts/modules.sh > /dev/null

echo "--- RUNNING ---"
date

./scripts/venv/bin/python3 ./scripts/measure_operation_breakdown.py "--num_iter $NUM_ITER" -only_measure_total

echo ""

echo "--- DONE ---"
date