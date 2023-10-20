
#include "device_minors.cuh"
#include <iostream>
#include <stdio.h>

__global__ void minors2xN_kernel(int *minors_out, int *matrix, size_t N)
{
    auto col1 = blockIdx.x * blockDim.x + threadIdx.x;
    auto col2 = blockIdx.y * blockDim.y + threadIdx.y;
    printf("\n===\ncol1: %d col2: %d", col1, col2);
    if (col2 > col1 || col1 >= N || col2 >= N)
    {
        printf("@@exceeds matrix size\n");
        return;
    }
    printf("\n");
    printf("to index: %d\n", idx(col1, col2, N));
    minors_out[idx(col1, col2, N)] = minor2x2_2xN(matrix, col1, col2, N);
}

void minors2xN_cpu(int *minors_out, int *matrix, size_t N)
{
    for (size_t col1 = 0; col1 < N; col1++)
    {
        for (size_t col2 = col1 + 1; col2 < N; col2++)
        {
            size_t index = idx(col1, col2, N);
            std::cout << "\n===\ncol1: " << col1 << " col2: " << col2 << "\n";
            minors_out[index] = minor2x2_2xN(matrix, col1, col2, N);
            std::cout << minors_out[index] << " ";
        }
    }
}

void minors2xN_device(int numBlocks, int threadsPerBlock, thrust::device_vector<int> minors_out, thrust::device_vector<int> matrix, size_t N)
{
    minors2xN_kernel<<<numBlocks, threadsPerBlock>>>(
        thrust::raw_pointer_cast(minors_out.data()),
        thrust::raw_pointer_cast(matrix.data()),
        N);
}

void minors2xN_cpu(thrust::host_vector<int> minors_out, thrust::host_vector<int> matrix, size_t N)
{
    minors2xN_cpu(
        thrust::raw_pointer_cast(minors_out.data()),
        thrust::raw_pointer_cast(matrix.data()),
        N);
}