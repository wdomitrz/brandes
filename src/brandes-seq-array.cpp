#include <cstdint>
#include <iostream>
#include <vector>

#include "brandes.hpp"
#define add_to_Q(x) \
    { Q[Q_size++] = x; }
#define add_to_S(x) \
    { S[S_size++] = x; }

void brandes_kernel(const int32_t n, const int32_t starting_positions[],
                    const int32_t compact_graph[], double CB[], int32_t* S,
                    int32_t* sigma, int32_t* d, int32_t* Q, double* delta);

void brandes(const int32_t n, const int32_t starting_positions[],
             const int32_t compact_graph[], double CB[]) {
    std::vector<int32_t> S(n);
    std::vector<int32_t> sigma(n);
    std::vector<int32_t> d(n);
    std::vector<int32_t> Q(starting_positions[n]);
    std::vector<double> delta(n);
    brandes_kernel(n, starting_positions, compact_graph, CB, S.data(),
                   sigma.data(), d.data(), Q.data(), delta.data());
}

void brandes_kernel(const int32_t n, const int32_t starting_positions[],
                    const int32_t compact_graph[], double CB[], int32_t* S,
                    int32_t* sigma, int32_t* d, int32_t* Q, double* delta) {
    for (int32_t s = 0; s < n; s++) {
        int32_t S_size = 0;
        std::vector<std::vector<int32_t>> P(n);
        std::fill_n(sigma, n, 0);
        sigma[s] = 1;
        std::fill_n(d, n, -1);
        d[s] = 0;
        size_t Q_size = 0;
        add_to_Q(s);
        for (size_t Q_pos = 0; Q_pos < Q_size; Q_pos++) {
            int32_t v = Q[Q_pos];
            add_to_S(v);
            int32_t end = starting_positions[v + 1];
            for (int32_t i = starting_positions[v]; i < end; i++) {
                int32_t w = compact_graph[i];
                if (d[w] < 0) {
                    add_to_Q(w);
                    d[w] = d[v] + 1;
                }
                if (d[w] == d[v] + 1) {
                    sigma[w] += sigma[v];
                    P[w].push_back(v);
                }
            }
        }
        std::fill_n(delta, n, 0.0);
        for (size_t i = S_size; i > 0; i--) {
            int32_t w = S[i - 1];

            for (size_t i = 0; i < P[w].size(); i++) {
                int32_t v = P[w][i];
                if (sigma[w] != 0) {
                    delta[v] += ((double)sigma[v]) / ((double)sigma[w]) *
                                ((double)1.0 + (double)delta[w]);
                }
            }
            if (w != s) {
                CB[w] += delta[w];
            }
        }
    }
}
