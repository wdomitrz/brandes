for t in brandes-seq brandes-seq-vector brandes-seq-array brandes-par-vert brandes-par-edge brandes-par-vert-queue brandes-par-vert-comp brandes-par-edge-comp brandes-par-vert-queue-comp brandes-par-vert-comp-virt brandes-par-vert-comp-virt-stride brandes-par-vert-comp-virt-stride-const; do
    echo $t
    rm -f ./brandes
    make $t
    make test
done
