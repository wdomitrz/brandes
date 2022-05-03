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
        const std::vector<std::pair<node_id, node_id>>& edges) {
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

    inline const node_id* get_starting_positions_of_nodes() {
        return starting_positions_of_nodes.data();
    }

    inline const node_id* get_compact_graph() { return compact_graph.data(); }

    inline node_id size() { return number_of_nodes; }
};
class Compacted_graph_representation {
   private:
    Compact_graph_representation whole_graph;
    std::vector<int32_t> reach;
    std::vector<int32_t> deg;
    std::vector<double> CB;
    std::vector<bool> removed;
    std::vector<int32_t> id_in_small_graph;
    std::vector<int32_t> reach_compacted;
    Compact_graph_representation* small_graph;

    void remove(int32_t s) {
        // assert(deg[s] == 1);
        while (s != -1) {
            removed[s] = true;
            int32_t parent = -1;
            for (int32_t i = whole_graph.get_starting_positions_of_nodes()[s];
                 i < whole_graph.get_starting_positions_of_nodes()[s + 1];
                 i++) {
                const int32_t u = whole_graph.get_compact_graph()[i];
                if (!removed[u]) {
                    parent = u;
                    break;
                }
            }
            deg[parent]--;
            reach[parent] += reach[s];
            CB[parent] += ((double)2) * ((double)reach[s]) *
                          ((double)whole_graph.size() - reach[parent]);
            if (deg[parent] == 1) {
                s = parent;
            } else {
                s = -1;
            }
        }
    }

    void initialize_degrees() {
        deg.resize(whole_graph.size());
        for (int32_t s = 0; s < whole_graph.size(); s++) {
            deg[s] = whole_graph.get_starting_positions_of_nodes()[s + 1] -
                     whole_graph.get_starting_positions_of_nodes()[s];
        }
    }
    void remove_nodes() {
        CB.resize(whole_graph.size(), 0);
        reach.resize(whole_graph.size(), 1);
        removed.resize(whole_graph.size(), false);
        for (int32_t s = 0; s < whole_graph.size(); s++) {
            if (!removed[s] && deg[s] == 1) remove(s);
        }
    }

    void compute_ids_and_compacted_reach() {
        id_in_small_graph.resize(whole_graph.size());
        int32_t next_id = 0;
        for (int32_t s = 0; s < whole_graph.size(); s++) {
            if (!removed[s]) {
                id_in_small_graph[s] = next_id;
                reach_compacted.push_back(reach[s]);
                next_id++;
            }
        }
    }

    std::vector<std::pair<int32_t, int32_t>> get_compacted_edges(
        const std::vector<std::pair<int32_t, int32_t>>& edges) {
        std::vector<std::pair<int32_t, int32_t>> compacted_edges;
        for (const auto& node_1_node_2 : edges) {
            if (!removed[node_1_node_2.first] &&
                !removed[node_1_node_2.second]) {
                compacted_edges.emplace_back(
                    id_in_small_graph[node_1_node_2.first],
                    id_in_small_graph[node_1_node_2.first]);
            }
        }
        return compacted_edges;
    }

   public:
    Compacted_graph_representation(
        const std::vector<std::pair<int32_t, int32_t>>& edges)
        : whole_graph(edges) {
        initialize_degrees();
        remove_nodes();
        compute_ids_and_compacted_reach();
        small_graph =
            new Compact_graph_representation(get_compacted_edges(edges));
    }
    const int32_t* get_reach() { return reach_compacted.data(); }
    const Compact_graph_representation get_whole_compace() {
        return whole_graph;
    }
    const int32_t* get_starting_positions_of_nodes() {
        return small_graph->get_starting_positions_of_nodes();
    }

    const int32_t* get_compact_graph() {
        return small_graph->get_compact_graph();
    }

    int32_t size() { return small_graph->size(); }

    std::vector<double> centrality_for_original_graph(
        const std::vector<double> centrality_for_small_graph) {
        for (int32_t s = 0; s < whole_graph.size(); s++) {
            if (!removed[s]) {
                CB[s] += centrality_for_small_graph[id_in_small_graph[s]];
            }
        }
        return CB;
    }
};
