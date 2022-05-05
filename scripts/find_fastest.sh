#!/usr/bin/env bash

for name in brandes-par-vert-virt-stride brandes-par-vert-virt brandes-par-vert-comp-virt-stride brandes-par-vert-comp-virt-stride-const brandes-par-vert-comp-virt brandes-par-vert-queue-comp brandes-par-edge-comp brandes-par-vert-comp brandes-par-vert-queue brandes-par-edge brandes-par-vert; do
    echo "$name"
    ~/srun-run -w steven bash -c "make \"$name\" > /dev/null"
    ~/srun-run -w steven time ./brandes ../tests/loc-gowalla_edges.txt ./res/res-loc-gowalla-"$name".txt
    rm -f ./brandes
    echo ""
done
