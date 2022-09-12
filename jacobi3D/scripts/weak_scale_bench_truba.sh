#!/bin/bash

#SBATCH -J stencil-bench-weak
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -A proj16
#SBATCH -p palamut-cuda
#SBATCH --gres=gpu:8
#SBATCH --time=6:00:00
#SBATCH -o stencil_bench_weak_output_%j.log

. ./scripts/modules_truba.sh > /dev/null

MAX_NUM_GPUS=8
CUDA_VISIBLE_DEVICES_SETTING=("0" "0" "0,1" "0,1,2" "0,1,2,3" "0,1,2,3,4" "0,1,2,3,4,5" "0,1,2,3,4,5,6" "0,1,2,3,4,5,6,7" )

declare -A version_name_to_idx_map

version_name_to_idx_map["Baseline Copy"]=0
version_name_to_idx_map["Baseline Copy Overlap"]=1
version_name_to_idx_map["Baseline P2P"]=2
version_name_to_idx_map["Baseline Single Copy"]=3

version_name_to_idx_map["Single Stream 1TB"]=4
version_name_to_idx_map["Single Stream 2TB"]=5
version_name_to_idx_map["Double Stream"]=6

version_name_to_idx_map["Baseline Copy (No compute)"]=7
version_name_to_idx_map["Baseline Copy Overlap (No Compute)"]=8
version_name_to_idx_map["Baseline P2P (No Compute)"]=9

version_name_to_idx_map["Single Stream 1TB (No Compute)"]=10
version_name_to_idx_map["Single Stream 2TB (No Compute)"]=11
version_name_to_idx_map["Double Stream (No Compute)"]=12

BIN="./jacobi -s 1"

STARTING_NX=${STARTING_NX:-256}
STARTING_NY=${STARTING_NY:-256}
STARTING_NZ=${STARTING_NZ:-256}

NUM_ITER=${NUM_ITER:-100000}
NUM_RUNS=${NUM_RUNS:-5}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi

  shift
done

for version_name in "${!version_name_to_idx_map[@]}"; do
    echo "Running ${version_name}"; echo ""

    version_idx=${version_name_to_idx_map[$version_name]}

    NX=${STARTING_NX}
    NY=${STARTING_NY}
    NZ=${STARTING_NZ}

    for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS*=2 )); do
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}

        echo "Num GPUS: ${NUM_GPUS}"
        echo "${NUM_ITER} iterations on grid ${NX}x${NY}x${NZ}"

        for (( i=1; i <= ${NUM_RUNS}; i++ )); do
            execution_time=$(${BIN} -v ${version_idx} -nx ${NX} -ny ${NY} -nz  ${NZ} -niter ${NUM_ITER})
            echo "${execution_time} on run ${i}"
        done

        printf "\n"

        NZ=$((2*NZ))
       
    done

    echo "-------------------------------------"
done