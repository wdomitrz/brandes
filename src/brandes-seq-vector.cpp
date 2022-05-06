#include <cstdint>
#include <iostream>
#include <vector>

#include "brandes-old.hpp"

void brandes(const uint32_t n, const uint32_t starting_positions[],
             const uint32_t compact_graph[], double CB[]) {
    for (uint32_t s = 0; s < n; s++) {
        std::vector<uint32_t> S;
        std::vector<std::vector<uint32_t>> P(n);
        std::vector<uint32_t> sigma(n, 0);
        sigma[s] = 1;
        std::vector<uint32_t> d(n, UINT32_MAX);
        d[s] = 0;
        std::vector<uint32_t> Q;
        Q.push_back(s);
        for (size_t Q_pos = 0; Q_pos < Q.size(); Q_pos++) {
            uint32_t v = Q[Q_pos];
            S.push_back(v);
            uint32_t end = starting_positions[v + 1];
            for (uint32_t i = starting_positions[v]; i < end; i++) {
                uint32_t w = compact_graph[i];
                if (d[w] < 0) {
                    Q.push_back(w);
                    d[w] = d[v] + 1;
                }
                if (d[w] == d[v] + 1) {
                    sigma[w] += sigma[v];
                    P[w].push_back(v);
                }
            }
        }
        std::vector<double> delta(n, 0.0);
        for (size_t i = S.size(); i > 0; i--) {
            uint32_t w = S[i - 1];

            for (size_t i = 0; i < P[w].size(); i++) {
                uint32_t v = P[w][i];
                delta[v] += ((double)sigma[v]) / ((double)sigma[w]) *
                            ((double)1.0 + (double)delta[w]);
            }
            if (w != s) {
                CB[w] += delta[w];
            }
        }
    }
}
