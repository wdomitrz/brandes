#include <cuda.h>

#include <cstdint>
#include <cstring>
#include <iostream>

#include "brandes.hpp"
#include "errors.hpp"
#include "sizes.hpp"

__global__ void brandes_kernel(const uint32_t n,
                               const uint32_t starting_positions[],
                               const uint32_t compact_graph[],
                               const uint32_t reach[], double CB[],
                               uint32_t* sigma, uint32_t* d, double* delta);

void brandes(const uint32_t n, const uint32_t starting_positions[],
             const uint32_t compact_graph[], const uint32_t reach[],
             double CB[]) {
    if (n == 0 || starting_positions[n] == 0) return;
    uint32_t *starting_positions_dev, *reach_dev, *compact_graph_dev, *sigma,
        *d;
    double *delta, *CB_dev;
    HANDLE_ERROR(cudaMalloc((void**)&starting_positions_dev,
                            sizeof(uint32_t) * (n + 1)));
    HANDLE_ERROR(cudaMalloc((void**)&reach_dev, sizeof(uint32_t) * n));
    HANDLE_ERROR(cudaMalloc((void**)&compact_graph_dev,
                            sizeof(uint32_t) * starting_positions[n]));
    HANDLE_ERROR(cudaMalloc((void**)&CB_dev, sizeof(double) * n));
    HANDLE_ERROR(cudaMalloc((void**)&sigma, sizeof(uint32_t) * n * BLOCKS));
    HANDLE_ERROR(cudaMalloc((void**)&d, sizeof(uint32_t) * n * BLOCKS));
    HANDLE_ERROR(cudaMalloc((void**)&delta, sizeof(double) * n * BLOCKS));
    HANDLE_ERROR(cudaMemcpy(starting_positions_dev, starting_positions,
                            sizeof(uint32_t) * (n + 1),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(reach_dev, reach, sizeof(uint32_t) * n,
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(compact_graph_dev, compact_graph,
                            sizeof(uint32_t) * starting_positions[n],
                            cudaMemcpyHostToDevice));
    // HANDLE_ERROR(cudaMemset(CB_res, 0.0, sizeof(double) * n));
    brandes_kernel<<<BLOCKS, THREADS>>>(n, starting_positions_dev,
                                        compact_graph_dev, reach_dev, CB_dev,
                                        sigma, d, delta);
    HANDLE_ERROR(
        cudaMemcpy(CB, CB_dev, sizeof(double) * n, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(delta));
    HANDLE_ERROR(cudaFree(d));
    HANDLE_ERROR(cudaFree(sigma));
    HANDLE_ERROR(cudaFree(CB_dev));
    HANDLE_ERROR(cudaFree(compact_graph_dev));
    HANDLE_ERROR(cudaFree(reach_dev));
    HANDLE_ERROR(cudaFree(starting_positions_dev));
}

__global__ void brandes_kernel(const uint32_t n,
                               const uint32_t starting_positions[],
                               const uint32_t compact_graph[],
                               const uint32_t reach[], double CB[],
                               uint32_t* sigma_global, uint32_t* d_global,
                               double* delta_global) {
    const uint32_t my_start = threadIdx.x;
    const uint32_t my_end = n;
    const uint32_t my_step = blockDim.x;
    __shared__ bool cont;
    __shared__ uint32_t l;
    __shared__ uint32_t* sigma;
    __shared__ uint32_t* d;
    __shared__ double* delta;
    if (threadIdx.x == 0) {
        sigma = &sigma_global[n * blockIdx.x];
        d = &d_global[n * blockIdx.x];
        delta = &delta_global[n * blockIdx.x];
    }
    if (blockIdx.x == 0)
        for (uint32_t i = my_start; i < my_end; i += my_step) {
            CB[i] = 0;
        }
    for (uint32_t s = blockIdx.x; s < n; s += gridDim.x) {
        __syncthreads();
        for (uint32_t i = my_start; i < my_end; i += my_step) {
            sigma[i] = 0;
            d[i] = UINT32_MAX;
            delta[i] = reach[i];
        }
        __syncthreads();
        if (my_start == 0) {
            sigma[s] = 1;
            d[s] = 0;
            cont = true;
            l = 0;
        }
        __syncthreads();
        while (cont) {
            __syncthreads();
            cont = false;
            __syncthreads();
            for (uint32_t u = my_start; u < my_end; u += my_step) {
                if (d[u] == l) {
                    const uint32_t end = starting_positions[u + 1];
                    for (uint32_t i = starting_positions[u]; i < end; i++) {
                        const uint32_t v = compact_graph[i];
                        if (d[v] == UINT32_MAX) {
                            d[v] = l + 1;
                            cont = true;
                        }
                        if (d[v] == l + 1) {
                            atomicAdd(&sigma[v], sigma[u]);
                        }
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                l++;
            }
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
                        if (d[v] == l + 1) {
                            delta[u] +=
                                (double)sigma[u] / (double)sigma[v] * delta[v];
                        }
                    }
                }
            }
        }
        __syncthreads();
        for (uint32_t v = my_start; v < my_end; v += my_step) {
            if (v != s) {
                atomicAdd(&CB[v],
                          (double)reach[s] * (delta[v] - (double)reach[v]));
            }
        }
    }
}
