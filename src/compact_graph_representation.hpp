#include <cstdint>
#include <vector>

using node_id = int32_t;
class Compact_graph_representation {
    // With a guardian at the end
    std::vector<node_id> starting_positions_of_nodes;
    std::vector<node_id> compact_graph;
    node_id number_of_nodes;

   private:
    void add_directed_edge(std::pair<node_id, node_id> node_1_node_2,
                           std::vector<node_id>& considered_edges_from_node) {
        compact_graph[starting_positions_of_nodes[node_1_node_2.first] +
                      considered_edges_from_node[node_1_node_2.first]] =
            node_1_node_2.second;
        considered_edges_from_node[node_1_node_2.first]++;
    }

   public:
    Compact_graph_representation(
        std::vector<std::pair<node_id, node_id>>& edges) {
        node_id max_node_id = 0;
        for (const auto& node_1_node_2 : edges)
            if (node_1_node_2.second > max_node_id)
                max_node_id = node_1_node_2.second;

        number_of_nodes = max_node_id + 1;
        starting_positions_of_nodes.resize(number_of_nodes + 1, 0);
        compact_graph.resize(2 * edges.size());

        for (const auto& node_1_node_2 : edges) {
            starting_positions_of_nodes[node_1_node_2.first]++;
            starting_positions_of_nodes[node_1_node_2.second]++;
        }

        // be optimized, but not worth it -- not a bootle neck at all
        for (std::size_t i = 1; i < starting_positions_of_nodes.size(); i++) {
            starting_positions_of_nodes[i] +=
                starting_positions_of_nodes[i - 1];
        }
        for (std::size_t i = starting_positions_of_nodes.size() - 1; i > 0;
             i--) {
            starting_positions_of_nodes[i] = starting_positions_of_nodes[i - 1];
        }
        starting_positions_of_nodes[0] = 0;

        std::vector<node_id> considered_edges_from_node(number_of_nodes, 0);
        for (const auto& node_1_node_2 : edges) {
            add_directed_edge(node_1_node_2, considered_edges_from_node);
            add_directed_edge(
                std::make_pair(node_1_node_2.second, node_1_node_2.first),
                considered_edges_from_node);
        }
    }

    const node_id* get_starting_positions_of_nodes() {
        return starting_positions_of_nodes.data();
    }

    const node_id* get_compact_graph() { return compact_graph.data(); }

    node_id size() { return number_of_nodes; }
};
