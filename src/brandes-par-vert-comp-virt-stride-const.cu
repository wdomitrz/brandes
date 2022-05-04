#include <cuda.h>

#include <cstdint>
#include <cstring>
#include <iostream>

#include "brandes-virt-stride.hpp"
#include "errors.hpp"
#include "sizes.hpp"

__constant__ int32_t n, virt_n;

__global__ void brandes_kernel(const int32_t starting_positions[],
                               const int32_t compact_graph[],
                               const int32_t reach[], const int32_t vmap[],
                               const int32_t vptrs[], const int32_t jmp[],
                               double CB[], int32_t* sigma, int32_t* d,
                               double* delta);

void brandes(const int32_t host_n, const int32_t host_virt_n,
             const int32_t starting_positions[], const int32_t compact_graph[],
             const int32_t reach[], const int32_t vmap[], const int32_t vptrs[],
             const int32_t jmp[], double CB[]) {
    if (host_n == 0 || starting_positions[host_n] == 0) return;
    int32_t *starting_positions_dev, *reach_dev, *compact_graph_dev, *vmap_dev,
        *vptrs_dev, *jmp_dev, *sigma, *d;
    double *delta, *CB_dev;
    HANDLE_ERROR(cudaMemcpyToSymbol(n, &host_n, sizeof(int32_t)));
    HANDLE_ERROR(cudaMemcpyToSymbol(virt_n, &host_virt_n, sizeof(int32_t)));
    HANDLE_ERROR(cudaMalloc((void**)&starting_positions_dev,
                            sizeof(int32_t) * (host_n + 1)));
    HANDLE_ERROR(cudaMalloc((void**)&reach_dev, sizeof(int32_t) * host_n));
    HANDLE_ERROR(cudaMalloc((void**)&jmp_dev, sizeof(int32_t) * host_n));
    HANDLE_ERROR(cudaMalloc((void**)&compact_graph_dev,
                            sizeof(int32_t) * starting_positions[host_n]));
    HANDLE_ERROR(cudaMalloc((void**)&vmap_dev, sizeof(int32_t) * host_virt_n));
    HANDLE_ERROR(
        cudaMalloc((void**)&vptrs_dev, sizeof(int32_t) * (host_virt_n + 1)));
    HANDLE_ERROR(cudaMalloc((void**)&CB_dev, sizeof(double) * host_n));
    HANDLE_ERROR(cudaMalloc((void**)&sigma, sizeof(int32_t) * host_n * BLOCKS));
    HANDLE_ERROR(cudaMalloc((void**)&d, sizeof(int32_t) * host_n * BLOCKS));
    HANDLE_ERROR(cudaMalloc((void**)&delta, sizeof(double) * host_n * BLOCKS));
    HANDLE_ERROR(cudaMemcpy(starting_positions_dev, starting_positions,
                            sizeof(int32_t) * (host_n + 1),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(reach_dev, reach, sizeof(int32_t) * host_n,
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(vmap_dev, vmap, sizeof(int32_t) * host_virt_n,
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(vptrs_dev, vptrs,
                            sizeof(int32_t) * (host_virt_n + 1),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(jmp_dev, jmp, sizeof(int32_t) * host_n,
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(compact_graph_dev, compact_graph,
                            sizeof(int32_t) * starting_positions[host_n],
                            cudaMemcpyHostToDevice));
    // HANDLE_ERROR(cudaMemset(CB_res, 0.0, sizeof(double) * host_n));
    brandes_kernel<<<BLOCKS, THREADS>>>(
        starting_positions_dev, compact_graph_dev, reach_dev, vmap_dev,
        vptrs_dev, jmp_dev, CB_dev, sigma, d, delta);
    HANDLE_ERROR(cudaMemcpy(CB, CB_dev, sizeof(double) * host_n,
                            cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(delta));
    HANDLE_ERROR(cudaFree(d));
    HANDLE_ERROR(cudaFree(sigma));
    HANDLE_ERROR(cudaFree(CB_dev));
    HANDLE_ERROR(cudaFree(compact_graph_dev));
    HANDLE_ERROR(cudaFree(reach_dev));
    HANDLE_ERROR(cudaFree(vmap_dev));
    HANDLE_ERROR(cudaFree(vptrs_dev));
    HANDLE_ERROR(cudaFree(jmp_dev));
    HANDLE_ERROR(cudaFree(starting_positions_dev));
}

__global__ void brandes_kernel(const int32_t starting_positions[],
                               const int32_t compact_graph[],
                               const int32_t reach[], const int32_t vmap[],
                               const int32_t vptrs[], const int32_t jmp[],
                               double CB[], int32_t* sigma_global,
                               int32_t* d_global, double* delta_global) {
    const int32_t my_start = threadIdx.x;
    const int32_t my_end = n;
    const int32_t my_step = blockDim.x;
    __shared__ bool cont;
    __shared__ int32_t l;
    __shared__ int32_t* sigma;
    __shared__ int32_t* d;
    __shared__ double* delta;
    if (threadIdx.x == 0) {
        sigma = &sigma_global[n * blockIdx.x];
        d = &d_global[n * blockIdx.x];
        delta = &delta_global[n * blockIdx.x];
    }
    if (blockIdx.x == 0)
        for (int i = my_start; i < my_end; i += my_step) {
            CB[i] = 0;
        }
    for (int32_t s = blockIdx.x; s < n; s += gridDim.x) {
        __syncthreads();
        for (int i = my_start; i < my_end; i += my_step) {
            sigma[i] = 0;
            d[i] = -1;
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
            for (int32_t u_virt = my_start; u_virt < virt_n;
                 u_virt += my_step) {
                const int32_t u = vmap[u_virt];
                if (d[u] == l) {
                    const int32_t end = starting_positions[u + 1];
                    const int32_t now_jmp = jmp[u];
                    for (int32_t i = vptrs[u_virt]; i < end; i += now_jmp) {
                        const int32_t v = compact_graph[i];
                        if (d[v] == -1) {
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
            for (int32_t u_virt = my_start; u_virt < virt_n;
                 u_virt += my_step) {
                const int32_t u = vmap[u_virt];
                if (d[u] == l) {
                    double sum = 0;
                    const int32_t end = starting_positions[u + 1];
                    const int32_t now_jmp = jmp[u];
                    for (int32_t i = vptrs[u_virt]; i < end; i += now_jmp) {
                        const int32_t v = compact_graph[i];
                        if (d[v] == l + 1) {
                            sum +=
                                (double)sigma[u] / (double)sigma[v] * delta[v];
                        }
                    }
                    atomicAdd(&delta[u], sum);
                }
            }
        }
        __syncthreads();
        for (int32_t v = my_start; v < my_end; v += my_step) {
            if (v != s) {
                atomicAdd(&CB[v],
                          (double)reach[s] * (delta[v] - (double)reach[v]));
            }
        }
    }
}
