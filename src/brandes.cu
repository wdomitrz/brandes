#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "compact_graph_representation.hpp"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Two arguments are required\n"
                     "./brandes input-file output-file\n";
        exit(1);
    }
    const std::string input_file_name(argv[1]), out_file_name(argv[2]);
    std::ifstream input_file(input_file_name);
    std::vector<std::pair<node_id, node_id>> edges;

    node_id node_1, node_2;
    while (input_file >> node_1 >> node_2) {
        edges.push_back(std::make_pair(node_1, node_2));
    }

    Compact_graph_representation graph(edges);

    node_id n = graph.size();
    const node_id *compact_graph = graph.get_compact_graph();
    const node_id *starting_positions = graph.get_starting_positions_of_nodes();

    for (int i = 0; i < n; i++) {
        for (int j = starting_positions[i]; j < starting_positions[i + 1];
             j++) {
            std::cout << i << " " << compact_graph[j] << "\n";
        }
    }
}
