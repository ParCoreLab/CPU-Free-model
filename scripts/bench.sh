#!/bin/bash

MAX_NUM_GPUS=8
CUDA_VISIBLE_DEVICES_SETTING=("0" "0" "0,1" "0,1,2" "0,1,2,3" "0,1,2,3,4" "0,1,2,3,4,5" "0,1,2,3,4,5,6" "0,1,2,3,4,5,6,7" )

VERSIONS=(
    "Single Stream Multi Threaded"
    "Single Stream Multi Threaded 2TB"
    "Single Stream Single Threaded"
    "Single Stream Single Threaded 2TB"
)

STARTING_NX=256
STARTING_NY=256
NUM_ITER=1000000
NUM_RUNS=5

EXECUTABLE=$1

for version_idx in ${!VERSIONS[@]}; do
    echo "Running ${VERSIONS[version_idx]}"; echo ""

    NX=${STARTING_NX}
    NY=${STARTING_NY}

    for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS*=2 )); do
        echo "Num GPUS: ${NUM_GPUS}"
        echo "${NUM_ITER} iterations on grid ${NX}x${NY}"
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}

        for (( i=1; i <= ${NUM_RUNS}; i++ )); do
            execution_time=$(${EXECUTABLE} -v ${version_idx} -nx ${NX} -ny ${NY} -niter ${NUM_ITER})
            echo "${execution_time} on run ${i}"
        done

        if [[ $NX -le $NY ]]; then
            NX=$((2*NX))
        else
            NY=$((2*NY))
        fi

        echo ""
    done

    echo "-------------------------------------"
done

