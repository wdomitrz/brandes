#!/usr/bin/env python3
import sys

import networkx as nx


def main():
    if len(sys.argv) != 3:
        print("Wrong number of arguments")
        exit(1)

    with open(sys.argv[1]) as in_file:
        data = in_file.read().splitlines()
    edges = list(map(lambda x: list(map(int, x.split())), data))
    n = max(map(max, edges)) + 1
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    betwenness = nx.centrality.betweenness_centrality(G, normalized=False)
    res = [str(2 * v) for v in betwenness.values()]

    with open(sys.argv[2], "w+") as out_file:
        out_file.write("\n".join(res) + "\n")


if __name__ == "__main__":
    main()
