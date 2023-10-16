#include <iostream>
#include "kernels.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main()
{
    size_t n = 1 << 3;
    float *x, *y, *z;
    cudaMallocManaged(&x, n * sizeof(int));
    cudaMallocManaged(&y, n * sizeof(int));
    cudaMallocManaged(&z, n * sizeof(int));

    for (int i = 0; i < n; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add<<<n, 1>>>(x, y, z, n);

    cudaDeviceSynchronize();

    for (int i = 0; i < n; i++)
    {
        std::cout << z[i] << std::endl;
    }

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
}