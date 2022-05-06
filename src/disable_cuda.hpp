const struct zxcv { uint32_t x = 0; } posX;
const struct asd { uint32_t x = 1; } sizeX;
#define __global__
#define __shared__
#define __syncthreads() \
    {}
#define atomicAdd(x, y) \
    { *x += y; }
#define threadIdx posX
#define blockDim sizeX
#define blockIdx posX
#define gridDim sizeX
#define HANDLE_ERROR(x) \
    { x }
#define cudaMalloc(x, size) \
    { *(x) = (void*)malloc(size); }
#define cudaFree(x) \
    { free(x); }
#define cudaMemcpy(dest, src, size, how) \
    { std::memcpy(dest, src, size); }
