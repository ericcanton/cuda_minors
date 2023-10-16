#include <iostream>
#include "kernels.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

void cpu_minors(int *mat, int *out, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        int *minor = new int[(m - 1) * (n - 1)];
        for (int j = 0; j < n; j++)
        {
            int minor_i = 0;
            for (int k = 0; k < m; k++)
            {
                if (k == i)
                    continue;
                for (int l = 0; l < n; l++)
                {
                    if (l == j)
                        continue;
                    minor[minor_i++] = mat[k * n + l];
                }
            }
            out[i * n + j] = minor[0] * minor[3] - minor[1] * minor[2];
        }
    }
}

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