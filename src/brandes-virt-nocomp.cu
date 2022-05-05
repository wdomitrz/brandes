#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "brandes-virt-nocomp.hpp"
#include "compact_graph_representation.hpp"
#include "sizes.hpp"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Two arguments are required\n"
                     "./brandes in-file out-file\n";
        exit(1);
    }
    const std::string in_file_name(argv[1]), out_file_name(argv[2]);
    std::ifstream in_file(in_file_name);
    std::vector<std::pair<int32_t, int32_t>> edges;

    int32_t node_1, node_2;
    while (in_file >> node_1 >> node_2) {
        if (node_1 < node_2) edges.emplace_back(node_1, node_2);
    }

    Compact_graph_representation whole_graph(edges);
    Virtualized_graph_representation<Compact_graph_representation> virt_graph(
        whole_graph, MDEG);

    int32_t n = virt_graph.orig_size(), virt_n = virt_graph.size();
    const int32_t *compact_graph = virt_graph.get_compact_graph();
    const int32_t *starting_positions =
        virt_graph.get_starting_positions_of_nodes();
    const int32_t *vmap = virt_graph.get_vmap();
    const int32_t *vptrs = virt_graph.get_vptrs();

    std::vector<double> res(n, 0);
    brandes(n, virt_n, starting_positions, compact_graph, vmap, vptrs,
            res.data());

    std::ofstream out_file(out_file_name);
    for (size_t i = 0; i < res.size(); i++) {
        out_file << res[i] << "\n";
    }
}
