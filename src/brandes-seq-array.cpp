#include <cstdint>
#include <iostream>
#include <vector>

#include "brandes.hpp"
#define add_to_Q(x) \
    { Q[Q_size++] = x; }
#define add_to_S(x) \
    { S[S_size++] = x; }
#define add_to_P(x, y) \
    { P[starting_positions[x] + P_pos[x]++] = y; }

void brandes_kernel(const int32_t n, const int32_t starting_positions[],
                    const int32_t compact_graph[], double CB[], int32_t* S,
                    int32_t* sigma, int32_t* d, int32_t* Q, double* delta,
                    int32_t* P, int32_t* P_pos);

void brandes(const int32_t n, const int32_t starting_positions[],
             const int32_t compact_graph[], double CB[]) {
    std::vector<int32_t> S(n);
    std::vector<int32_t> sigma(n);
    std::vector<int32_t> d(n);
    std::vector<int32_t> Q(starting_positions[n]);
    std::vector<double> delta(n);
    std::vector<int32_t> P(starting_positions[n]);
    std::vector<int32_t> P_pos(n);
    brandes_kernel(n, starting_positions, compact_graph, CB, S.data(),
                   sigma.data(), d.data(), Q.data(), delta.data(), P.data(),
                   P_pos.data());
}

void brandes_kernel(const int32_t n, const int32_t starting_positions[],
                    const int32_t compact_graph[], double CB[], int32_t* S,
                    int32_t* sigma, int32_t* d, int32_t* Q, double* delta,
                    int32_t* P, int32_t* P_pos) {
    for (int32_t s = 0; s < n; s++) {
        std::fill_n(sigma, n, 0);
        std::fill_n(d, n, -1);
        std::fill_n(delta, n, 0.0);
        std::fill_n(P_pos, n, 0);

        int32_t S_size = 0;
        int32_t Q_size = 0;
        sigma[s] = 1;
        d[s] = 0;

        add_to_Q(s);
        for (int32_t Q_pos = 0; Q_pos < Q_size; Q_pos++) {
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
                    add_to_P(w, v);
                }
            }
        }
        for (int32_t i = S_size; i > 0; i--) {
            int32_t w = S[i - 1];

            int32_t P_iter_end = starting_positions[w] + P_pos[w];
            for (int32_t i = starting_positions[w]; i < P_iter_end; i++) {
                int32_t v = P[i];
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
