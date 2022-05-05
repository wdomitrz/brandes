#!/usr/bin/env bash

for THREADS in 1024 768 512 384 256 128; do
    for BLOCKS in 2048 1024 3072 512 3584; do
        for MDEG in 3 4 6 8 10 12 14 16; do
            echo $THREADS $BLOCKS $MDEG
            rm -f ./brandes
            echo $THREADS $BLOCKS $MDEG >times/meas-$THREADS-$BLOCKS-$MDEG.out
            ~/srun-run -w steven bash -c "nvcc --compiler-options -Wextra --compiler-options -Wall --compiler-options -O3 -o brandes src/brandes-virt-stride.cu src/brandes-par-vert-comp-virt-stride.cu -arch=sm_61 -DTHREADS=$THREADS -DBLOCKS=$BLOCKS -DMDEG=$MDEG"
            ~/srun-run -w steven bash -c "{ time ./brandes ../tests/loc-gowalla_edges.txt res.txt; } 2>>times/meas-$THREADS-$BLOCKS-$MDEG.out"
        done
    done
done
