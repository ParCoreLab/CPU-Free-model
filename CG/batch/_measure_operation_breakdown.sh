#!/bin/bash

#SBATCH --job-name=cg-bench
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --partition=dgx2q
#SBATCH --time=06:00:00
#SBATCH --output=sbatch_output_%j.log

cd ~/multi-perks/CG

. ./scripts/modules.sh > /dev/null

echo "--- RUNNING ---"
date

./scripts/venv/bin/python3 ./scripts/measure_operation_breakdown.py

echo ""

echo "--- DONE ---"
date