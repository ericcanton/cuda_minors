#pragma once

#include <string>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cuda_runtime.h"

#define BLOCK_SIZE 32
#define idx(i, j, N) ((i) * (N) + (j))
#define minor2x2_2xN(mat, i, j, N) (mat[idx(i, 0, N)] * mat[idx(j, 1, N)] - mat[idx(i, 1, N)] * mat[idx(j, 0, N)])

__global__ void minors2xN_kernel(int *minors_out, int *matrix, size_t N);
void minors2xN_device(int numBlocks, int threadsPerBlock, thrust::device_vector<int> minors_out, thrust::device_vector<int> matrix, size_t N);

void minors2xN_cpu(int *minors_out, int *matrix, size_t N);
void minors2xN_cpu(thrust::host_vector<int> minors_out, thrust::host_vector<int> matrix, size_t N);