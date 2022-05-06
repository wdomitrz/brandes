#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "brandes-virt-stride.hpp"
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
    std::vector<std::pair<uint32_t, uint32_t>> edges;

    uint32_t node_1, node_2;
    while (in_file >> node_1 >> node_2) {
        if (node_1 < node_2) edges.emplace_back(node_1, node_2);
    }

    Compact_graph_representation whole_graph(edges);
    Virtualized_graph_representation_with_stride<Compacted_graph_representation>
        virt_graph(whole_graph, MDEG);

    uint32_t n = virt_graph.orig_size(), virt_n = virt_graph.size();
    const uint32_t *compact_graph = virt_graph.get_compact_graph();
    const uint32_t *starting_positions =
        virt_graph.get_starting_positions_of_nodes();
    const uint32_t *reach = virt_graph.get_reach();
    const uint32_t *vmap = virt_graph.get_vmap();
    const uint32_t *vptrs = virt_graph.get_vptrs();
    const uint32_t *jmp = virt_graph.get_jmp();

    std::vector<double> res_small(n, 0);
    brandes(n, virt_n, starting_positions, compact_graph, reach, vmap, vptrs,
            jmp, res_small.data());
    std::vector<double> res =
        virt_graph.centrality_for_original_graph(res_small);

    std::ofstream out_file(out_file_name);
    for (size_t i = 0; i < res.size(); i++) {
        out_file << res[i] << "\n";
    }
}
