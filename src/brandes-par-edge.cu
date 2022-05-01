#include <cuda.h>

#include <cstdint>
#include <cstring>
#include <iostream>

#include "brandes.hpp"
#include "errors.hpp"

__global__ void brandes_kernel(const int32_t n,
                               const int32_t starting_positions[],
                               const int32_t compact_graph[], double CB[],
                               int32_t* sigma, int32_t* d, double* delta);

void brandes(const int32_t n, const int32_t starting_positions[],
             const int32_t compact_graph[], double CB[]) {
    int32_t *starting_positions_dev, *compact_graph_dev, *sigma, *d,
        *compact_graph_ext;
    double *delta, *CB_dev;
    compact_graph_ext =
        (int32_t*)malloc(sizeof(int32_t) * 2 * starting_positions[n]);
    for (int32_t i = 0; i < n; i++)
        for (int32_t j = starting_positions[i]; j < starting_positions[i + 1];
             j++) {
            compact_graph_ext[2 * j] = i;
            compact_graph_ext[2 * j + 1] = compact_graph[j];
        }

    HANDLE_ERROR(
        cudaMalloc((void**)&starting_positions_dev, sizeof(int32_t) * (n + 1)));
    HANDLE_ERROR(cudaMalloc((void**)&compact_graph_dev,
                            sizeof(int32_t) * 2 * starting_positions[n]));
    HANDLE_ERROR(cudaMalloc((void**)&CB_dev, sizeof(double) * n));
    HANDLE_ERROR(cudaMalloc((void**)&sigma, sizeof(int32_t) * n));
    HANDLE_ERROR(cudaMalloc((void**)&d, sizeof(int32_t) * n));
    HANDLE_ERROR(cudaMalloc((void**)&delta, sizeof(double) * n));
    HANDLE_ERROR(cudaMemcpy(starting_positions_dev, starting_positions,
                            sizeof(int32_t) * (n + 1), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(compact_graph_dev, compact_graph_ext,
                            sizeof(int32_t) * 2 * starting_positions[n],
                            cudaMemcpyHostToDevice));
    // HANDLE_ERROR(cudaMemset(CB_dev, 0.0, sizeof(double) * n));
    brandes_kernel<<<1, 1024>>>(n, starting_positions_dev, compact_graph_dev,
                                CB_dev, sigma, d, delta);
    HANDLE_ERROR(
        cudaMemcpy(CB, CB_dev, sizeof(double) * n, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(delta));
    HANDLE_ERROR(cudaFree(d));
    HANDLE_ERROR(cudaFree(sigma));
    HANDLE_ERROR(cudaFree(CB_dev));
    HANDLE_ERROR(cudaFree(compact_graph_dev));
    HANDLE_ERROR(cudaFree(starting_positions_dev));
    free(compact_graph_ext);
}

__global__ void brandes_kernel(const int32_t n,
                               const int32_t starting_positions[],
                               const int32_t compact_graph[], double CB[],
                               int32_t* sigma, int32_t* d, double* delta) {
    const int32_t my_start = threadIdx.x + blockIdx.x * blockDim.x;
    const int32_t my_end = starting_positions[n];
    const int32_t my_step = blockDim.x * gridDim.x;
    const int32_t my_start_n = threadIdx.x + blockIdx.x * blockDim.x;
    const int32_t my_end_n = n;
    const int32_t my_step_n = blockDim.x * gridDim.x;
    __shared__ bool cont;
    __shared__ int32_t l;
    for (int i = my_start_n; i < my_end_n; i += my_step_n) {
        CB[i] = 0;
    }
    for (int32_t s = 0; s < n; s++) {
        __syncthreads();
        for (int i = my_start_n; i < my_end_n; i += my_step_n) {
            sigma[i] = 0;
            d[i] = -1;
            delta[i] = 0.0;
        }
        __syncthreads();
        if (my_start == 0) {
            sigma[s] = 1;
            d[s] = 0;
            cont = true;
            l = 0;
        }
        __syncthreads();
        do {
            __syncthreads();
            cont = false;
            __syncthreads();
            for (int32_t i = my_start; i < my_end; i += my_step) {
                const int32_t u = compact_graph[2 * i];
                const int32_t v = compact_graph[2 * i + 1];
                if (d[u] == l) {
                    if (d[v] == -1) {
                        d[v] = l + 1;
                        cont = true;
                    }
                    if (d[v] == l + 1) {
                        atomicAdd(&sigma[v], sigma[u]);
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                l++;
            }
        } while (cont);
        __syncthreads();
        for (; l > 1;) {
            __syncthreads();
            if (threadIdx.x == 0) l--;
            __syncthreads();
            for (int32_t i = my_start; i < my_end; i += my_step) {
                const int32_t u = compact_graph[2 * i];
                const int32_t v = compact_graph[2 * i + 1];
                if (d[u] == l && d[v] - 1 == d[u]) {
                    atomicAdd(&delta[u], ((double)sigma[u]) /
                                             ((double)sigma[v]) *
                                             ((double)1.0 + (double)delta[v]));
                }
            }
        }

        __syncthreads();
        for (int32_t v = my_start; v < n; v += my_step) {
            if (v != s) {
                CB[v] += delta[v];
            }
        }
    }
}
