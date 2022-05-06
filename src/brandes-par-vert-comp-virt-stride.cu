#include <cuda.h>

#include <cstdint>
#include <cstring>
#include <iostream>

#include "brandes-virt-stride.hpp"
#include "errors.hpp"
#include "sizes.hpp"

#define allign_up_to_ALLIGN
#define allign_up_to_ALLIGN_dev
// #define ALLIGN 32

// inline size_t allign_up_to_ALLIGN_dev(size_t x) {
//     if (x % ALLIGN == 0)
//         return x;
//     else
//         return (x + (x - (x % ALLIGN)));
// }

// __device__ inline size_t allign_up_to_ALLIGN_dev(size_t x) {
//     if (x % ALLIGN == 0)
//         return x;
//     else
//         return (x + (x - (x % ALLIGN)));
// }

__global__ void brandes_kernel(const uint32_t n, const uint32_t virt_n,
                               const uint32_t starting_positions[],
                               const uint32_t compact_graph[],
                               const uint32_t reach[], const uint32_t vmap[],
                               const uint32_t vptrs[], const uint32_t jmp[],
                               double CB[], uint32_t* sigma, uint32_t* d,
                               double* delta);

__global__ void collect_CB(const size_t n, double CB[], const size_t end) {
    for (size_t which = blockIdx.x * blockDim.x + threadIdx.x; which < n;
         which += blockDim.x * gridDim.x) {
        for (size_t i = 1; i < end; i++) {
            CB[which] += CB[which + n * i];
        }
    }
}

void brandes(const uint32_t n, const uint32_t virt_n,
             const uint32_t starting_positions[],
             const uint32_t compact_graph[], const uint32_t reach[],
             const uint32_t vmap[], const uint32_t vptrs[],
             const uint32_t jmp[], double CB[]) {
    if (n == 0 || starting_positions[n] == 0) {
        std::cerr << 0 << "\n" << 0 << "\n";
        return;
    }
    uint32_t *starting_positions_dev, *reach_dev, *compact_graph_dev, *vmap_dev,
        *vptrs_dev, *jmp_dev, *d;
    uint32_t* sigma;
    double *delta, *CB_dev;
    cudaStream_t stream[6];
    for (size_t i = 0; i < 6; i++) {
        cudaStreamCreate(&stream[i]);
    }
    cudaEvent_t start_kernel, stop_kernel, start_with_memory, stop_with_memory;
    HANDLE_ERROR(cudaEventCreate(&start_with_memory));
    HANDLE_ERROR(cudaEventCreate(&start_kernel));
    HANDLE_ERROR(cudaEventCreate(&stop_kernel));
    HANDLE_ERROR(cudaEventCreate(&stop_with_memory));
    HANDLE_ERROR(cudaMalloc((void**)&starting_positions_dev,
                            sizeof(uint32_t) * (n + 1)));
    HANDLE_ERROR(cudaMalloc((void**)&reach_dev, sizeof(uint32_t) * n));
    HANDLE_ERROR(cudaMalloc((void**)&jmp_dev, sizeof(uint32_t) * n));
    HANDLE_ERROR(cudaMalloc((void**)&compact_graph_dev,
                            sizeof(uint32_t) * starting_positions[n]));
    HANDLE_ERROR(cudaMalloc((void**)&vmap_dev, sizeof(uint32_t) * virt_n));
    HANDLE_ERROR(
        cudaMalloc((void**)&vptrs_dev, sizeof(uint32_t) * (virt_n + 1)));
    HANDLE_ERROR(cudaMalloc((void**)&CB_dev, sizeof(double) * n * BLOCKS));
    HANDLE_ERROR(cudaMalloc((void**)&sigma, sizeof(uint32_t) * n * BLOCKS));
    HANDLE_ERROR(cudaMalloc((void**)&d, sizeof(uint32_t) * n * BLOCKS));
    HANDLE_ERROR(cudaMalloc((void**)&delta, sizeof(double) * n * BLOCKS));
    HANDLE_ERROR(cudaEventRecord(start_with_memory, 0));
    HANDLE_ERROR(cudaMemcpyAsync(starting_positions_dev, starting_positions,
                                 sizeof(uint32_t) * (n + 1),
                                 cudaMemcpyHostToDevice, stream[0]));
    HANDLE_ERROR(cudaMemcpyAsync(reach_dev, reach, sizeof(uint32_t) * n,
                                 cudaMemcpyHostToDevice, stream[1]));
    HANDLE_ERROR(cudaMemcpyAsync(vmap_dev, vmap, sizeof(uint32_t) * virt_n,
                                 cudaMemcpyHostToDevice, stream[2]));
    HANDLE_ERROR(cudaMemcpyAsync(vptrs_dev, vptrs,
                                 sizeof(uint32_t) * (virt_n + 1),
                                 cudaMemcpyHostToDevice, stream[3]));
    HANDLE_ERROR(cudaMemcpyAsync(jmp_dev, jmp, sizeof(uint32_t) * n,
                                 cudaMemcpyHostToDevice, stream[4]));
    HANDLE_ERROR(cudaMemcpyAsync(compact_graph_dev, compact_graph,
                                 sizeof(uint32_t) * starting_positions[n],
                                 cudaMemcpyHostToDevice, stream[5]));
    HANDLE_ERROR(cudaEventRecord(start_kernel, 0));
    brandes_kernel<<<BLOCKS, THREADS, 0, 0>>>(
        n, virt_n, starting_positions_dev, compact_graph_dev, reach_dev,
        vmap_dev, vptrs_dev, jmp_dev, CB_dev, sigma, d, delta);
    HANDLE_ERROR(cudaEventRecord(stop_kernel, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop_kernel));
    collect_CB<<<BLOCKS, THREADS, 0, 0>>>(n, CB_dev, BLOCKS);
    HANDLE_ERROR(
        cudaMemcpy(CB, CB_dev, sizeof(double) * n, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaEventRecord(stop_with_memory, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop_with_memory));

    float time_kernel, time_with_memory;
    HANDLE_ERROR(cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel));
    HANDLE_ERROR(cudaEventElapsedTime(&time_with_memory, start_with_memory,
                                      stop_with_memory));

    std::cerr << (unsigned long long)time_kernel << "\n"
              << (unsigned long long)time_with_memory << "\n";

    HANDLE_ERROR(cudaEventDestroy(start_with_memory));
    HANDLE_ERROR(cudaEventDestroy(start_kernel));
    HANDLE_ERROR(cudaEventDestroy(stop_kernel));
    HANDLE_ERROR(cudaEventDestroy(stop_with_memory));

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

__global__ void brandes_kernel(const uint32_t n, const uint32_t virt_n,
                               const uint32_t starting_positions[],
                               const uint32_t compact_graph[],
                               const uint32_t reach[], const uint32_t vmap[],
                               const uint32_t vptrs[], const uint32_t jmp[],
                               double CB_global[], uint32_t* sigma_global,
                               uint32_t* d_global, double* delta_global) {
    // const uint32_t big_step = 1 + (n - 1) / blockDim.x;
    // const uint32_t my_start = threadIdx. * big_step;
    // const uint32_t my_end = min(n, (threadIdx.x + 1) * big_step);
    // const uint32_t my_step = 1;
    const uint32_t my_start = threadIdx.x;
    const uint32_t my_end = n;
    const uint32_t my_end_virt = virt_n;
    const uint32_t my_step = blockDim.x;
    __shared__ bool cont;
    __shared__ uint32_t l;
    __shared__ uint32_t* sigma;
    __shared__ uint32_t* d;
    __shared__ double* delta;
    __shared__ double* CB;
    if (my_start == 0) {
        sigma = &sigma_global[n * blockIdx.x];
        d = &d_global[n * blockIdx.x];
        delta = &delta_global[n * blockIdx.x];
        CB = &CB_global[n * blockIdx.x];
    }
    __syncthreads();
    for (uint32_t i = my_start; i < my_end; i += my_step) {
        CB[i] = 0;
    }
    for (uint32_t s = blockIdx.x; s < my_end; s += gridDim.x) {
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
            for (uint32_t u_virt = my_start; u_virt < my_end_virt;
                 u_virt += my_step) {
                const uint32_t u = vmap[u_virt];
                if (d[u] == l) {
                    const uint32_t end = starting_positions[u + 1];
                    const uint32_t now_jmp = jmp[u];
                    for (uint32_t i = vptrs[u_virt]; i < end; i += now_jmp) {
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
            if (my_start == 0) {
                l++;
            }
        }
        __syncthreads();
        while (l > 1) {
            __syncthreads();
            if (my_start == 0) l--;
            __syncthreads();
            for (uint32_t u_virt = my_start; u_virt < my_end_virt;
                 u_virt += my_step) {
                const uint32_t u = vmap[u_virt];
                if (d[u] == l) {
                    double sum = 0;
                    const uint32_t end = starting_positions[u + 1];
                    const uint32_t now_jmp = jmp[u];
                    for (uint32_t i = vptrs[u_virt]; i < end; i += now_jmp) {
                        const uint32_t v = compact_graph[i];
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
        for (uint32_t v = my_start; v < my_end; v += my_step) {
            if (v != s) {
                CB[v] += (double)reach[s] * (delta[v] - (double)reach[v]);
            }
        }
    }
}
