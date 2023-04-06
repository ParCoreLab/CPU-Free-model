#!/bin/bash

#SBATCH -J stencil-bench-strong
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -c 16
#SBATCH -A proj16
#SBATCH -p palamut-cuda
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH -o stencil_bench_strong_output_%j.log

. ./scripts/modules_truba.sh > /dev/null

MAX_NUM_GPUS=8
CUDA_VISIBLE_DEVICES_SETTING=("0" "0" "0,1" "0,1,2" "0,1,2,3" "0,1,2,3,4" "0,1,2,3,4,5" "0,1,2,3,4,5,6" "0,1,2,3,4,5,6,7" )

declare -A version_name_to_idx_map

version_name_to_idx_map["Baseline Copy"]=0
version_name_to_idx_map["Baseline Copy Overlap"]=1
version_name_to_idx_map["Baseline P2P"]=2
#version_name_to_idx_map["Baseline Single Copy"]=3

version_name_to_idx_map["Single Stream 1TB"]=4
version_name_to_idx_map["Single Stream 2TB"]=5
version_name_to_idx_map["Double Stream"]=6

version_name_to_idx_map["Baseline Copy (No compute)"]=7
version_name_to_idx_map["Baseline Copy Overlap (No Compute)"]=8
version_name_to_idx_map["Baseline P2P (No Compute)"]=9

version_name_to_idx_map["Single Stream 1TB (No Compute)"]=10
version_name_to_idx_map["Single Stream 2TB (No Compute)"]=11
version_name_to_idx_map["Double Stream (No Compute)"]=12

declare -A version_name_to_idx_map_nvshmem

version_name_to_idx_map_nvshmem["NVSHMEM Baseline"]=0
version_name_to_idx_map_nvshmem["NVSHMEM Baseline Optimized"]=1

version_name_to_idx_map_nvshmem["NVSHMEM Single Stream 1TB"]=2
version_name_to_idx_map_nvshmem["NVSHMEM Single Stream 2TB"]=3
version_name_to_idx_map_nvshmem["NVSHMEM Double Stream"]=4

version_name_to_idx_map_nvshmem["NVSHMEM Baseline (No Compute)"]=5
version_name_to_idx_map_nvshmem["NVSHMEM Baseline Optimized (No Compute)"]=6

version_name_to_idx_map_nvshmem["NVSHMEM Single Stream 1TB (No Compute)"]=7
version_name_to_idx_map_nvshmem["NVSHMEM Single Stream 2TB (No Compute)"]=8
version_name_to_idx_map_nvshmem["NVSHMEM Double Stream (No Compute)"]=9


BIN="./jacobi -s 1"
NV_BIN="./jacobi_nvshmem -s 1"


NUM_ITER=${NUM_ITER:-10000}
NUM_RUNS=${NUM_RUNS:-5}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi

  shift
done


for (( STARTING_NX=16384; STARTING_NX<=16384; STARTING_NX*=2 )); do

    NX=${STARTING_NX}
    NY=${NX}

    for version_name in "${!version_name_to_idx_map[@]}"; do
        echo "Running ${version_name}"; echo ""

        version_idx=${version_name_to_idx_map[$version_name]}

        for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS+=1 )); do
            export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}

            echo "Num GPUS: ${NUM_GPUS}"
            echo "${NUM_ITER} iterations on grid ${NX}x${NY}"

            for (( i=1; i <= ${NUM_RUNS}; i++ )); do
                execution_time=$(${BIN} -v ${version_idx} -nx ${NX} -ny ${NY} -niter ${NUM_ITER})
                echo "${execution_time} on run ${i}"
            done

            printf "\n"
            
        done

        echo "-------------------------------------"
    done


    for version_name in "${!version_name_to_idx_map_nvshmem[@]}"; do
        echo "Running ${version_name}"; echo ""

        version_idx=${version_name_to_idx_map_nvshmem[$version_name]}

        for (( NP=1; NP <= ${MAX_NUM_GPUS}; NP+=1 )); do

            echo "Num GPUS: ${NP}"
            echo "${NUM_ITER} iterations on grid ${NX}x${NY}"

            for (( i=1; i <= ${NUM_RUNS}; i++ )); do
                execution_time=$(mpirun -np ${NP} ${NV_BIN} -v ${version_idx} -nx ${NX} -ny ${NY} -niter ${NUM_ITER})
                echo "${execution_time} on run ${i}"
            done

            printf "\n"

        done

        echo "-------------------------------------"
    done

    echo "#####################################" 
done