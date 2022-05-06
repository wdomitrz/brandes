#!/usr/bin/env bash

for THREADS in 1024 ; do
    for BLOCKS in 2048 1024 512 128 64 32 12 10 8; do
        for MDEG in 4 5 6 7 8 9 10 11 12; do
            echo $THREADS $BLOCKS $MDEG
            rm -f ./brandes
            echo $THREADS $BLOCKS $MDEG >times/meas-$THREADS-$BLOCKS-$MDEG.out
            ~/srun-run -w steven bash -c "nvcc --compiler-options -Wextra --compiler-options -Wall --compiler-options -O3 -o brandes src/brandes-virt-stride.cu src/brandes-par-vert-comp-virt-stride.cu -arch=sm_61 -DTHREADS=$THREADS -DBLOCKS=$BLOCKS -DMDEG=$MDEG"
            ~/srun-run -w steven bash -c "{ time ./brandes ../tests/loc-gowalla_edges.txt res.txt; } 2>>times/meas-$THREADS-$BLOCKS-$MDEG.out"
        done
    done
done
