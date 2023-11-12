#pragma once

#include <string>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cuda_runtime.h"

#define BLOCK_SIZE 32
// TODO: change this to column major if there's a reason/standard
// row major format
#define idxrm(r, c, nCol) (r * nCol + c)
#define minor_idx_2xN(col1, col2, N) (idxrm(col1, col2, N) - ((col1 + 2) * (col1 + 1) / 2))

__global__ void minors2xN_kernel(thrust::device_vector<int> *determinantsArray, thrust::device_vector<int> *matrixIn, size_t N);
void minors2xN_device(int numBlocks, int threadsPerBlock, thrust::device_vector<int> *minors_out, thrust::device_vector<int> *matrix, size_t N);

void minors2xN_cpu(thrust::host_vector<int> *minors_out, thrust::host_vector<int> *matrix, size_t N);