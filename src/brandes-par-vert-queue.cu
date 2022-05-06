#include <cuda.h>

#include <cstdint>
#include <cstring>
#include <iostream>

#include "brandes-old.hpp"
#include "errors.hpp"
#include "sizes.hpp"

__global__ void brandes_kernel(const uint32_t n,
                               const uint32_t starting_positions[],
                               const uint32_t compact_graph[], double CB[],
                               uint32_t* sigma, uint32_t* d, double* delta,
                               uint32_t* Q);

void brandes(const uint32_t n, const uint32_t starting_positions[],
             const uint32_t compact_graph[], double CB[]) {
    uint32_t *starting_positions_dev, *compact_graph_dev, *sigma, *d, *Q;
    double *delta, *CB_dev;
    HANDLE_ERROR(cudaMalloc((void**)&starting_positions_dev,
                            sizeof(uint32_t) * (n + 1)));
    HANDLE_ERROR(cudaMalloc((void**)&compact_graph_dev,
                            sizeof(uint32_t) * starting_positions[n]));
    HANDLE_ERROR(cudaMalloc((void**)&CB_dev, sizeof(double) * n));
    HANDLE_ERROR(cudaMalloc((void**)&Q, sizeof(uint32_t) * 2 * n * BLOCKS));
    HANDLE_ERROR(cudaMalloc((void**)&sigma, sizeof(uint32_t) * n * BLOCKS));
    HANDLE_ERROR(cudaMalloc((void**)&d, sizeof(uint32_t) * n * BLOCKS));
    HANDLE_ERROR(cudaMalloc((void**)&delta, sizeof(double) * n * BLOCKS));
    HANDLE_ERROR(cudaMemcpy(starting_positions_dev, starting_positions,
                            sizeof(uint32_t) * (n + 1),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(compact_graph_dev, compact_graph,
                            sizeof(uint32_t) * starting_positions[n],
                            cudaMemcpyHostToDevice));
    // HANDLE_ERROR(cudaMemset(CB_res, 0.0, sizeof(double) * n));
    brandes_kernel<<<BLOCKS, THREADS>>>(n, starting_positions_dev,
                                        compact_graph_dev, CB_dev, sigma, d,
                                        delta, Q);
    HANDLE_ERROR(
        cudaMemcpy(CB, CB_dev, sizeof(double) * n, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(delta));
    HANDLE_ERROR(cudaFree(d));
    HANDLE_ERROR(cudaFree(sigma));
    HANDLE_ERROR(cudaFree(CB_dev));
    HANDLE_ERROR(cudaFree(compact_graph_dev));
    HANDLE_ERROR(cudaFree(starting_positions_dev));
}

__global__ void brandes_kernel(const uint32_t n,
                               const uint32_t starting_positions[],
                               const uint32_t compact_graph[], double CB[],
                               uint32_t* sigma_global, uint32_t* d_global,
                               double* delta_global, uint32_t* Q_all_global) {
    const uint32_t my_start = threadIdx.x;
    const uint32_t my_end = n;
    const uint32_t my_step = blockDim.x;
    __shared__ uint32_t l;
    __shared__ uint32_t* sigma;
    __shared__ uint32_t* d;
    __shared__ double* delta;
    __shared__ uint32_t *Q, *next_Q, *Q_global;
    __shared__ uint32_t Q_size, next_Q_size;
    if (threadIdx.x == 0) {
        sigma = &sigma_global[n * blockIdx.x];
        d = &d_global[n * blockIdx.x];
        delta = &delta_global[n * blockIdx.x];
        Q_global = &Q_all_global[2 * n * blockIdx.x];
    }
    if (blockIdx.x == 0)
        for (int i = my_start; i < my_end; i += my_step) {
            CB[i] = 0;
        }
    for (uint32_t s = blockIdx.x; s < n; s += gridDim.x) {
        __syncthreads();
        for (int i = my_start; i < my_end; i += my_step) {
            sigma[i] = 0;
            d[i] = UINT32_MAX;
            delta[i] = 0.0;
        }
        __syncthreads();
        if (my_start == 0) {
            sigma[s] = 1;
            d[s] = 0;
            l = 0;
            Q = &Q_global[n * (l % 2)];
            next_Q = &Q_global[n * ((l + 1) % 2)];
            next_Q_size = 0;
            Q_size = 1;
            Q[0] = s;
        }
        __syncthreads();
        while (Q_size != 0) {
            __syncthreads();
            for (uint32_t j = my_start; j < Q_size; j += my_step) {
                const uint32_t u = Q[j];
                const uint32_t end = starting_positions[u + 1];
                for (uint32_t i = starting_positions[u]; i < end; i++) {
                    const uint32_t v = compact_graph[i];
                    if (atomicCAS(&d[v], UINT32_MAX, l + 1) == UINT32_MAX) {
                        // add to next Q
                        next_Q[atomicAdd(&next_Q_size, 1)] = v;
                    }
                    if (d[v] == l + 1) {
                        atomicAdd(&sigma[v], sigma[u]);
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                l++;
                Q = &Q_global[n * (l % 2)];
                next_Q = &Q_global[n * ((l + 1) % 2)];
                Q_size = next_Q_size;
                next_Q_size = 0;
            }
            __syncthreads();
        }
        __syncthreads();
        while (l > 1) {
            __syncthreads();
            if (threadIdx.x == 0) l--;
            __syncthreads();
            for (uint32_t u = my_start; u < my_end; u += my_step) {
                if (d[u] == l) {
                    const uint32_t end = starting_positions[u + 1];
                    for (uint32_t i = starting_positions[u]; i < end; i++) {
                        const uint32_t v = compact_graph[i];
                        if (d[v] - 1 == d[u]) {
                            delta[u] += ((double)sigma[u]) /
                                        ((double)sigma[v]) *
                                        ((double)1.0 + (double)delta[v]);
                        }
                    }
                }
            }
        }
        __syncthreads();
        for (uint32_t v = my_start; v < my_end; v += my_step) {
            if (v != s) {
                atomicAdd(&CB[v], delta[v]);
            }
        }
    }
}
