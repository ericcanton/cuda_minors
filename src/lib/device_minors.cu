
#include "device_minors.cuh"
#include <iostream>
#include <stdio.h>

__global__ void minors2xN_kernel(thrust::device_vector<int> *determinantsArray, thrust::device_vector<int> *matrixIn, size_t N)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    /*
determinant of
 {
    a0, a1,
    b0, b1,
 }
is
 a0 * b1 - a1 * b0
We use this for 2xN matrices by choosing 2 column indices i and j
where
 0 <= i < j <= N-1.
These (i, j) form entries in an upper triangular square matrix
[   -   (0, 1)  (0, 2)  ...     (0, N-1) ]
[   -     -     (1, 2)  ...     (1, N-1) ]
[   -     -       -     (2, 3) ...       ]
...
[   -     -       -     ...     (N-2, N-1) ]
of size (N-2) x (N-1). For N = 5, this looks like
-  01  02  03  04
-  -   12  13  14
-  -   -   23  24
-  -   -   -   34
where for example "13" means the determinant of the
2x2 submatrix formed by choosing columns 1 and 3 in:
 [  a0  a1  a2  a3  a4  ]
 [  b0  b1  b2  b3  b4  ].
That determinant is == ( a1 * b3 - a3 * b1 ).

For compactness, rather than putting this determinant in
 determinantsArray[idxrm(1, 3, 5)]
we want to cut out the dashes "-" in the matrix above.

The number of these to cut depends on i, the row index,
and is == ( (i + 2) * (i + 1) / 2 )

So, we arrive at: determinant for columns i and j, i < j,
 idxrm(i, j, N) -
is minor_idx_2xN(i, j, N) in "indexMath.h"
*/

    // determine i and j based on gid.x and gid.y
    // only do for 0 <= i < j < N
    if ((i >= j) || (j >= N))
        return;

    int a0 = (*matrixIn)[idxrm(0, i, N)];
    int a1 = (*matrixIn)[idxrm(0, j, N)];
    int b0 = (*matrixIn)[idxrm(1, i, N)];
    int b1 = (*matrixIn)[idxrm(1, j, N)];
    int det = (a0 * b1) - (b0 * a1);

    (*determinantsArray)[minor_idx_2xN(i, j, N)] = det;
}

void minors2xN_cpu(thrust::host_vector<int> *minors_out, thrust::host_vector<int> *matrix, size_t N)
{
    for (size_t col1 = 0; col1 < N; col1++)
    {
        for (size_t col2 = col1 + 1; col2 < N; col2++)
        {
            size_t index = minor_idx_2xN(col1, col2, N);
            int a0 = (*matrix)[idxrm(0, col1, N)];
            int a1 = (*matrix)[idxrm(0, col2, N)];
            int b0 = (*matrix)[idxrm(1, col1, N)];
            int b1 = (*matrix)[idxrm(1, col2, N)];
            (*minors_out)[index] = (a0 * b1) - (b0 * a1);
        }
    }
}

void minors2xN_device(int numBlocks, int threadsPerBlock, thrust::device_vector<int> *minors_out, thrust::device_vector<int> *matrix, size_t N)
{
    minors2xN_kernel<<<numBlocks, threadsPerBlock>>>(
        // thrust::raw_pointer_cast(minors_out.data()),
        // thrust::raw_pointer_cast(matrix.data()),
        minors_out,
        matrix,
        N);
}
