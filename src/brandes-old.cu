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

    Compact_graph_representation graph(edges);

    int32_t n = graph.size();
    const int32_t *compact_graph = graph.get_compact_graph();
    const int32_t *starting_positions = graph.get_starting_positions_of_nodes();

    for (int i = 0; i < n; i++) {
        for (int j = starting_positions[i]; j < starting_positions[i + 1];
             j++) {
        }
    }

    std::vector<double> res(n, 0);

    brandes(n, starting_positions, compact_graph, res.data());

    std::ofstream out_file(out_file_name);
    for (int32_t i = 0; i < n; i++) {
        out_file << res[i] << "\n";
    }
}
