
#include "device_minors.cuh"

__global__ void minors2byN_kernel(int *minors_out, int *matrix, size_t N)
{
    auto col1 = blockIdx.x * blockDim.x + threadIdx.x;
    auto col2 = blockIdx.y * blockDim.y + threadIdx.y;
    if (col2 > col1 || col1 >= N || col2 >= N)
        return;
    minors_out[idx(col1, col2, N)] = minor2x2_2xN(matrix, col1, col2, N);
}

void minors2byN_cpu(int *minors_out, int *matrix, size_t N)
{
    for (size_t col1 = 0; col1 < N; col1++)
    {
        for (size_t col2 = col1; col2 < N; col2++)
        {
            minors_out[idx(col1, col2, N)] = minor2x2_2xN(matrix, col1, col2, N);
        }
    }
}

__global__ void minors2byN_kernel(thrust::device_vector<int> minors_out, thrust::device_vector<int> matrix, size_t N)
{
    minors2byN_kernel(
        thrust::raw_pointer_cast(minors_out.data()),
        thrust::raw_pointer_cast(matrix.data()),
        N);
}

void minors2byN_cpu(thrust::device_vector<int> minors_out, thrust::device_vector<int> matrix, size_t N)
{
    minors2byN_cpu(
        thrust::raw_pointer_cast(minors_out.data()),
        thrust::raw_pointer_cast(matrix.data()),
        N);
}