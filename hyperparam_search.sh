#!/usr/bin/env bash

for THREADS in 1024 768 512 384 256 128; do
    for BLOCKS in 2048 2560 1536 1024 3072 512 3584; do
        for MDEG in 1 2 3 4 5 6 7 8; do
            rm ./brandes
            echo $THREADS $BLOCKS $MDEG >times/meas-$THREADS-$BLOCKS-$MDEG.out
            ~/srun-run -w steven bash -c "nvcc --compiler-options -Wextra --compiler-options -Wall --compiler-options -O3 -o brandes src/brandes-virt-stride.cu src/brandes-par-vert-comp-virt-stride.cu -arch=sm_61 -DTHREADS=$THREADS -DBLOCKS=$BLOCKS -DMDEG=$MDEG"
            ~/srun-run -w steven bash -c "{ time ./brandes ../tests/res-facebook_0.edges.txt res.txt; } 2>>times/meas-$THREADS-$BLOCKS-$MDEG.out"
        done
    done
done
