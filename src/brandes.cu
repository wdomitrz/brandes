#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "brandes.hpp"
#include "compact_graph_representation.hpp"

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
        edges.push_back(std::make_pair(node_1, node_2));
    }

    Compacted_graph_representation graph(edges);

    int32_t n = graph.size();
    const int32_t *compact_graph = graph.get_compact_graph();
    const int32_t *starting_positions = graph.get_starting_positions_of_nodes();
    const int32_t *reach = graph.get_reach();

    std::vector<double> res_small(n, 0);
    brandes(n, starting_positions, compact_graph, reach, res_small.data());
    std::vector<double> res = graph.centrality_for_original_graph(res_small);

    std::ofstream out_file(out_file_name);
    for (size_t i = 0; i < res.size(); i++) {
        out_file << res[i] << "\n";
    }
}
