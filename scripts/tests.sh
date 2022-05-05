#!/usr/bin/env bash
while true; do
    ./scripts/test.sh 20 0.2
    if [ $? -ne 0 ]; then break; fi
done
