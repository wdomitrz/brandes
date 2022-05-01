#include <cuda.h>

#include <cstdint>
#include <cstring>
#include <iostream>

#include "brandes.hpp"
#include "errors.hpp"
#define add_to_S(x) \
    { S[S_size++] = x; }
#define add_to_P(x, y) \
    { P[starting_positions[x] + P_pos[x]++] = y; }

__global__ void brandes_kernel(const int32_t n,
                               const int32_t starting_positions[],
                               const int32_t compact_graph[], double CB[],
                               int32_t* sigma, int32_t* d, double* delta,
                               int32_t* P, int32_t* P_pos);

void brandes(const int32_t n, const int32_t starting_positions[],
             const int32_t compact_graph[], double CB[]) {
    int32_t *starting_positions_dev, *compact_graph_dev, *sigma, *d, *P, *P_pos;
    double *delta, *CB_dev;
    HANDLE_ERROR(
        cudaMalloc((void**)&starting_positions_dev, sizeof(int32_t) * (n + 1)));
    HANDLE_ERROR(cudaMalloc((void**)&compact_graph_dev,
                            sizeof(int32_t) * starting_positions[n]));
    HANDLE_ERROR(cudaMalloc((void**)&CB_dev, sizeof(double) * n));
    HANDLE_ERROR(cudaMalloc((void**)&sigma, sizeof(int32_t) * n));
    HANDLE_ERROR(cudaMalloc((void**)&d, sizeof(int32_t) * n));
    HANDLE_ERROR(cudaMalloc((void**)&delta, sizeof(double) * n));
    HANDLE_ERROR(
        cudaMalloc((void**)&P, sizeof(int32_t) * starting_positions[n]));
    HANDLE_ERROR(cudaMalloc((void**)&P_pos, sizeof(int32_t) * n));
    HANDLE_ERROR(cudaMemcpy(starting_positions_dev, starting_positions,
                            sizeof(int32_t) * (n + 1), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(compact_graph_dev, compact_graph,
                            sizeof(int32_t) * starting_positions[n],
                            cudaMemcpyHostToDevice));
    // HANDLE_ERROR(cudaMemset(CB_dev, 0.0, sizeof(double) * n));
    brandes_kernel<<<1, 1>>>(n, starting_positions_dev, compact_graph_dev,
                             CB_dev, sigma, d, delta, P, P_pos);
    HANDLE_ERROR(
        cudaMemcpy(CB, CB_dev, sizeof(double) * n, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(P_pos));
    HANDLE_ERROR(cudaFree(P));
    HANDLE_ERROR(cudaFree(delta));
    HANDLE_ERROR(cudaFree(d));
    HANDLE_ERROR(cudaFree(sigma));
    HANDLE_ERROR(cudaFree(CB_dev));
    HANDLE_ERROR(cudaFree(compact_graph_dev));
    HANDLE_ERROR(cudaFree(starting_positions_dev));
}

__global__ void brandes_kernel(const int32_t n,
                               const int32_t starting_positions[],
                               const int32_t compact_graph[], double CB[],
                               int32_t* sigma, int32_t* d, double* delta,
                               int32_t* P, int32_t* P_pos) {
    const int32_t my_start = threadIdx.x + blockIdx.x * blockDim.x;
    const int32_t my_end = n;
    const int32_t my_step = blockDim.x * gridDim.x;
    __shared__ bool cont;
    __shared__ int32_t l;
    for (int i = my_start; i < my_end; i += my_step) {
        CB[i] = 0;
    }
    __syncthreads();
    for (int32_t s = 0; s < n; s++) {
        for (int i = my_start; i < my_end; i += my_step) {
            sigma[i] = 0;
            d[i] = -1;
            delta[i] = 0.0;
            P_pos[i] = 0;
        }
        __syncthreads();
        if (my_start == 0) {
            sigma[s] = 1;
            d[s] = 0;
            cont = true;
            l = 0;
        }
        __syncthreads();
        for (; cont; l++) {
            cont = false;
            __syncthreads();
            for (int32_t u = my_start; u < my_end; u += my_step) {
                if (d[u] == l) {
                    const int32_t end = starting_positions[u + 1];
                    for (int32_t i = starting_positions[u]; i < end; i++) {
                        const int32_t v = compact_graph[i];
                        if (d[v] == -1) {
                            d[v] = l + 1;
                            cont = true;
                        } else if (d[v] == l - 1) {
                            add_to_P(v, u);
                        }
                        if (d[v] == l + 1) {
                            atomicAdd(&sigma[v], sigma[u]);
                        }
                    }
                }
            }
        }
        for (; l > 1;) {
            l--;
            __syncthreads();
            for (int32_t u = my_start; u < my_end; u += my_step) {
                if (d[u] == l) {
                    int32_t P_iter_end = starting_positions[u] + P_pos[u];
                    for (int32_t i = starting_positions[u]; i < P_iter_end;
                         i++) {
                        const int32_t v = P[i];
                        if (sigma[u] != 0) {
                            delta[u] += ((double)sigma[u]) /
                                        ((double)sigma[v]) *
                                        ((double)1.0 + (double)delta[v]);
                        }
                    }
                }
            }
        }
        __syncthreads();
        for (int32_t v = my_start; v < my_end; v += my_step) {
            if (v != s) {
                CB[v] += delta[v];
            }
        }
    }
}
