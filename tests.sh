#!/usr/bin/env bash
while true; do
    ./test.sh 6 0.5
    if [ $? -ne 0 ]; then break; fi
done
