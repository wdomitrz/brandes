for t in brandes-par-vert-virt-stride brandes-par-vert-virt brandes-par-vert-comp-virt-stride brandes-par-vert-comp-virt-stride-const brandes-par-vert-comp-virt brandes-par-vert-queue-comp brandes-par-edge-comp brandes-par-vert-comp brandes-par-vert-queue brandes-par-edge brandes-par-vert brandes-seq-array brandes-seq-vector brandes-seq; do
    echo $t
    rm -f ./brandes
    make $t
    make test
done
