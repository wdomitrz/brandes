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

    const node_id* get_starting_positions_of_nodes() {
        return starting_positions_of_nodes.data();
    }

    const node_id* get_compact_graph() { return compact_graph.data(); }

    node_id size() { return number_of_nodes; }
};
class Compacted_graph_representation {
   private:
    Compact_graph_representation whole_graph;
    std::vector<int32_t> reach;
    std::vector<int32_t> deg;
    std::vector<int32_t> parent;
    std::vector<bool> removed;
    Compact_graph_representation* small_graph;

    void remove(int32_t v) {
        removed[v] = true;
        for (int32_t i = whole_graph.get_starting_positions_of_nodes()[v];
             i < whole_graph.get_starting_positions_of_nodes()[v + 1]; i++) {
            const int32_t u = whole_graph.get_compact_graph()[i];

            if (!removed[u]) {
                parent[v] = u;
                reach[u] += reach[v];
                deg[u]--;
                if (deg[u] == 1) remove(u);
            }
        }
    }

    std::vector<std::pair<int32_t, int32_t>> get_compacted_edges(
        const std::vector<std::pair<int32_t, int32_t>>& edges) {
        for (int32_t s = 0; s < whole_graph.size(); s++) {
            deg[s] = whole_graph.get_starting_positions_of_nodes()[s + 1] -
                     whole_graph.get_starting_positions_of_nodes()[s];
        }
        for (int32_t s = 0; s < whole_graph.size(); s++) {
            if (!removed[s] && deg[s] == 1) remove(s);
        }
        std::vector<std::pair<int32_t, int32_t>> compacted_edges;
        for (const auto& node_1_node_2 : edges) {
            if (!removed[node_1_node_2.first] &&
                !removed[node_1_node_2.second]) {
                compacted_edges.push_back(node_1_node_2);
            }
        }
        return compacted_edges;
    }

   public:
    Compacted_graph_representation(
        const std::vector<std::pair<int32_t, int32_t>>& edges)
        : whole_graph(edges) {
        reach.resize(whole_graph.size(), 0);
        deg.resize(whole_graph.size());
        parent.resize(whole_graph.size());
        removed.resize(whole_graph.size(), false);
        small_graph =
            new Compact_graph_representation(get_compacted_edges(edges));
    }
    // const int32_t* get_parent() { return parent.data(); }
    // const int32_t* get_reach() { return reach.data(); }
    // const bool* get_removed() { return removed.data(); }
    // const Compact_graph_representation get_whole_compace() {
    //     return whole_graph;
    // }
    // const int32_t* get_starting_positions_of_nodes() {
    //     return small_graph.get_starting_positions_of_nodes();
    // }

    // const int32_t* get_compact_graph() {
    //     return small_graph.get_compact_graph();
    // }

    // int32_t size() { return small_graph.size(); }
};
