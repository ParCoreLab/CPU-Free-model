#!/bin/bash

NUM_ITER=${NUM_ITER:-1000}
NUM_RUNS=${NUM_RUNS:-5}
FILENAME=${FILENAME:-USE_DEFAULT_FILENAME}
MATRICES_FOLDER=${MATRICES_FOLDER:-USE_DEFAULT_MATRICES_FOLDER}
GPU_MODEL=${GPU_MODEL:-A100}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi

  shift
done

WORK_DIR=~/multi-perks/CG
SCRATCH_DIR=/scratch/project/${PBS_ACCOUNT,,}/multi-perks-runs
# SCRATCH_DIR=/scratch/project/${PBS_ACCOUNT,,}/${USER}/multi-perks-runs

cd $WORK_DIR

. ./batch/Karolina/_load_karolina_modules.sh > /dev/null

cd $SCRATCH_DIR

cp $WORK_DIR/bin/cg ./bin/cg
cp $WORK_DIR/scripts/measure_single_gpu_runtime.py ./scripts/measure_single_gpu_runtime.py

echo "--- RUNNING ---"
date

python3 ./scripts/measure_single_gpu_runtime.py --num_iter $NUM_ITER --num_runs $NUM_RUNS --filename $FILENAME --matrices_folder $MATRICES_FOLDER --gpu_model $GPU_MODEL

echo ""

echo "--- DONE ---"
date

cd $WORK_DIR