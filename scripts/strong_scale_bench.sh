#!/bin/bash

#SBATCH --job-name=stencil-bench
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00

. ./scripts/modules.sh > /dev/null

MAX_NUM_GPUS=8
CUDA_VISIBLE_DEVICES_SETTING=("0" "0" "0,1" "0,1,2" "0,1,2,3" "0,1,2,3,4" "0,1,2,3,4,5" "0,1,2,3,4,5,6" "0,1,2,3,4,5,6,7" )

declare -A version_name_to_idx_map

version_name_to_idx_map["Single Stream Multi Threaded 1TB"]=0
version_name_to_idx_map["Single Stream Multi Threaded 2TB"]=1

version_name_to_idx_map["Baseline Multi Threaded Copy"]=3
version_name_to_idx_map["Baseline Multi Threaded Copy Overlap"]=4
version_name_to_idx_map["Baseline Multi Threaded P2P"]=5

version_name_to_idx_map["Single Stream Multi Threaded 1TB (No Compute)"]=9
version_name_to_idx_map["Single Stream Multi Threaded 2TB (No Compute)"]=10

BIN="./jacobi -s 1"

NX=${NX:-16384}
NY=${NY:-16384}
NUM_ITER=${NUM_ITER:-100000}
NUM_RUNS=${NUM_RUNS:-5}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        echo $1 $2 
   fi

  shift
done

for version_name in "${!version_name_to_idx_map[@]}"; do
    echo "Running ${version_name}"; echo ""

    version_idx=${version_name_to_idx_map[$version_name]}

    for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS+=1 )); do
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}

        echo "Num GPUS: ${NUM_GPUS}"
        echo "${NUM_ITER} iterations on grid ${NY}x${NX}"

        for (( i=1; i <= ${NUM_RUNS}; i++ )); do
            execution_time=$(${BIN} -v ${version_idx} -nx ${NX} -ny ${NY} -niter ${NUM_ITER})
            echo "${execution_time} on run ${i}"
        done

        printf "\n"
    done

    echo "-------------------------------------"
done
