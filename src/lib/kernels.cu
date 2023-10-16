
#include "kernels.cuh"

// CUDA kernel for vector addition
__global__ void add(float *a, float *b, float *c, size_t n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        c[index] = a[index] + b[index];
    }
}
