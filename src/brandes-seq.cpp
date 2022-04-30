#include <cstdint>
#include <iostream>
#include <list>
#include <queue>
#include <stack>
#include <vector>

#include "brandes.hpp"

void brandes(const int32_t n, const int32_t starting_positions[],
             const int32_t compact_graph[], double CB[]) {
    for (int s = 0; s < n; s++) {
        std::stack<int32_t> S;
        std::vector<std::vector<int32_t>> P(n);
        std::vector<int32_t> sigma(n, 0);
        sigma[s] = 1;
        std::vector<int32_t> d(n, -1);
        d[s] = 0;
        std::queue<int32_t> Q;
        Q.push(s);
        while (!Q.empty()) {
            int32_t v = Q.front();
            Q.pop();
            S.push(v);
            int32_t end = starting_positions[v + 1];
            for (int32_t i = starting_positions[v]; i < end; i++) {
                int32_t w = compact_graph[i];
                if (d[w] < 0) {
                    Q.push(w);
                    d[w] = d[v] + 1;
                }
                if (d[w] == d[v] + 1) {
                    sigma[w] += sigma[v];
                    P[w].push_back(v);
                }
            }
        }
        std::vector<double> delta(n, 0.0);
        while (!S.empty()) {
            int32_t w = S.top();
            S.pop();

            for (size_t i = 0; i < P[w].size(); i++) {
                int32_t v = P[w][i];
                if (sigma[w] != 0) {
                    delta[v] += ((double)sigma[v]) / ((double)sigma[w]) *
                                ((double)1.0 + (double)delta[w]);
                }
                if (w != s) {
                    CB[w] += delta[w];
                }
            }
        }
    }
}
