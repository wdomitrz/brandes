#!/usr/bin/env python3
import sys
from networkx import fast_gnp_random_graph

n = int(sys.argv[1])
p = float(sys.argv[2])

g = fast_gnp_random_graph(n, p)


for u, v in g.edges:
    print(u, v)
