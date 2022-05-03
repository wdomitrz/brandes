#!/usr/bin/env python3
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from verifyprograms import *

ref_result = read_brandes_scores(sys.argv[1])
prog_result = read_brandes_scores(sys.argv[2])
(comp, error) = compare_brandes_scores(ref_result, prog_result)
if not comp:
    print(f"error in the result: {error}"),
    exit(1)
else:
    exit(0)
