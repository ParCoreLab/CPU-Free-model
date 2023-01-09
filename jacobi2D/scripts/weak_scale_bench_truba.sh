#!/bin/bash

#SBATCH -J stencil-bench-weak
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -c 16
#SBATCH -A proj16
#SBATCH -p palamut-cuda
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH -o stencil_bench_weak_output_%j.log

. ./scripts/modules_truba.sh > /dev/null

MAX_NUM_GPUS=8
CUDA_VISIBLE_DEVICES_SETTING=("0" "0" "0,1" "0,1,2" "0,1,2,3" "0,1,2,3,4" "0,1,2,3,4,5" "0,1,2,3,4,5,6" "0,1,2,3,4,5,6,7" )

declare -A version_name_to_idx_map

#version_name_to_idx_map["Baseline Copy"]=0
version_name_to_idx_map["Baseline Copy Overlap"]=1
version_name_to_idx_map["Baseline P2P"]=2
#version_name_to_idx_map["Baseline Single Copy"]=3

version_name_to_idx_map["Single Stream 1TB Tile-by-Tile"]=4
version_name_to_idx_map["Single Stream 1TB Plane-by-Plane"]=5
#version_name_to_idx_map["Single Stream 2TB"]=6
#version_name_to_idx_map["Double Stream"]=7

#version_name_to_idx_map["Baseline Copy (No compute)"]=8
version_name_to_idx_map["Baseline Copy Overlap (No Compute)"]=9
version_name_to_idx_map["Baseline P2P (No Compute)"]=10

version_name_to_idx_map["Single Stream 1TB Tile-by-Tile (No Compute)"]=11
version_name_to_idx_map["Single Stream 1TB Tile-by-Tile (No Compute)"]=12
#version_name_to_idx_map["Single Stream 2TB Tile-by-Tile (No Compute)"]=13
#version_name_to_idx_map["Double Stream (No Compute)"]=14

declare -A version_name_to_idx_map_nvshmem

version_name_to_idx_map_nvshmem["Baseline NVSHMEM"]=15
version_name_to_idx_map_nvshmem["Baseline NVSHMEM Optimized"]=16

version_name_to_idx_map_nvshmem["NVSHMEM Single Stream 1TB Plane-by-Plane"]=17
#version_name_to_idx_map_nvshmem["NVSHMEM Single Stream 2TB Plane-by-Plane"]=18
#version_name_to_idx_map_nvshmem["NVSHMEM Double Stream Plane-by-Plane"]=19

version_name_to_idx_map_nvshmem["Baseline NVSHMEM (No Compute)"]=20
version_name_to_idx_map_nvshmem["Baseline NVSHMEM Optimized (No Compute)"]=21

version_name_to_idx_map_nvshmem["NVSHMEM Single Stream 1TB Plane-by-Plane (No Compute)"]=22

BIN="./jacobi -s 1"

MAX_NX=${MAX_NX:-256}
MAX_NY=${MAX_NY:-256}
MAX_NZ=${MAX_NZ:-256}

STARTING_NX=${STARTING_NX:-64}
STARTING_NY=${STARTING_NY:-64}
STARTING_NZ=${STARTING_NZ:-64}

NUM_ITER=${NUM_ITER:-100000}
NUM_RUNS=${NUM_RUNS:-5}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi

  shift
done


for (( NX = ${STARTING_NX}; NX <= ${MAX_NX}; NX*=2 )); do

    
    for version_name in "${!version_name_to_idx_map[@]}"; do
        echo "Running ${version_name}"; echo ""
        NY=${NX}
        NZ=${NX}
        version_idx=${version_name_to_idx_map[$version_name]}

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

    for version_name in "${!version_name_to_idx_map_nvshmem[@]}"; do
        echo "Running ${version_name}"; echo ""
        NY=${NX}
        NZ=${NX}
        version_idx=${version_name_to_idx_map_nvshmem[$version_name]}

        for (( NP=1; NP <= ${MAX_NUM_GPUS}; NP*=2 )); do

            echo "Num GPUS: ${NP}"
            echo "${NUM_ITER} iterations on grid ${NX}x${NY}x${NZ}"

            for (( i=1; i <= ${NUM_RUNS}; i++ )); do
                execution_time=$(mpirun -np ${NP} ./jacobi -s 1 -v ${version_idx} -nx ${NX} -ny ${NY} -nz  ${NZ} -niter ${NUM_ITER})
                echo "${execution_time} on run ${i}"
            done

            printf "\n"

            NZ=$((2*NZ))
        done

        echo "-------------------------------------"
    done
    
    echo "-------------------------------------"
    echo "-------------------------------------"
done
