for t in brandes-seq brandes-seq-vector brandes-seq-array brandes-par-vert brandes-par-edge brandes-par-vert-queue brandes-par-vert-comp; do
    echo $t
    rm -f ./brandes
    make $t
    make test
done
