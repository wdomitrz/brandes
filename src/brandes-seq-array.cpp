#include <cstdint>
#include <iostream>

#include "brandes-old.hpp"
#define add_to_Q(x) \
    { Q[Q_size++] = x; }
#define add_to_S(x) \
    { S[S_size++] = x; }
#define add_to_P(x, y) \
    { P[starting_positions[x] + P_pos[x]++] = y; }

void brandes_kernel(const uint32_t n, const uint32_t starting_positions[],
                    const uint32_t compact_graph[], double CB[], uint32_t* S,
                    uint32_t* sigma, uint32_t* d, uint32_t* Q, double* delta,
                    uint32_t* P, uint32_t* P_pos);

void brandes(const uint32_t n, const uint32_t starting_positions[],
             const uint32_t compact_graph[], double CB[]) {
    uint32_t* S = (uint32_t*)malloc(sizeof(uint32_t) * n);
    uint32_t* sigma = (uint32_t*)malloc(sizeof(uint32_t) * n);
    uint32_t* d = (uint32_t*)malloc(sizeof(uint32_t) * n);
    uint32_t* Q = (uint32_t*)malloc(sizeof(uint32_t) * starting_positions[n]);
    double* delta = (double*)malloc(sizeof(double) * n);
    uint32_t* P = (uint32_t*)malloc(sizeof(uint32_t) * starting_positions[n]);
    uint32_t* P_pos = (uint32_t*)malloc(sizeof(uint32_t) * n);
    brandes_kernel(n, starting_positions, compact_graph, CB, S, sigma, d, Q,
                   delta, P, P_pos);
    free(P_pos);
    free(P);
    free(delta);
    free(Q);
    free(d);
    free(sigma);
    free(S);
}

void brandes_kernel(const uint32_t n, const uint32_t starting_positions[],
                    const uint32_t compact_graph[], double CB[], uint32_t* S,
                    uint32_t* sigma, uint32_t* d, uint32_t* Q, double* delta,
                    uint32_t* P, uint32_t* P_pos) {
    for (uint32_t i = 0; i < n; i++) {
        CB[i] = 0.0;
    }
    for (uint32_t s = 0; s < n; s++) {
        for (uint32_t i = 0; i < n; i++) {
            sigma[i] = 0;
            d[i] = UINT32_MAX;
            delta[i] = 0.0;
            P_pos[i] = 0;
        }
        uint32_t S_size = 0;
        uint32_t Q_size = 0;
        sigma[s] = 1;
        d[s] = 0;

        add_to_Q(s);
        for (uint32_t Q_pos = 0; Q_pos < Q_size; Q_pos++) {
            uint32_t v = Q[Q_pos];
            add_to_S(v);
            uint32_t end = starting_positions[v + 1];
            for (uint32_t i = starting_positions[v]; i < end; i++) {
                uint32_t w = compact_graph[i];
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
        for (uint32_t i = S_size; i > 0; i--) {
            uint32_t w = S[i - 1];

            uint32_t P_iter_end = starting_positions[w] + P_pos[w];
            for (uint32_t i = starting_positions[w]; i < P_iter_end; i++) {
                uint32_t v = P[i];
                delta[v] += ((double)sigma[v]) / ((double)sigma[w]) *
                            ((double)1.0 + (double)delta[w]);
            }
            if (w != s) {
                CB[w] += delta[w];
            }
        }
    }
}
