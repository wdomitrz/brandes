#!/usr/bin/env bash

if [ $# -eq 2 ]; then
    ./scripts/gen.py $1 $2 >./test.txt
fi
./brandes ./test.txt out_my.txt
./scripts/brandes_test_gen.py ./test.txt out_ok.txt
./scripts/compare.py out_ok.txt out_my.txt
