#include <iostream>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "device_minors.cuh"

template <typename T>
void print_matrix(T A, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; ++j)
            std::cout << A[idxrm(i, j, n)] << " ";
        std::cout << std::endl;
    }
    // two "=" per element, plus one "=" per space between elements
    for (int i = 0; i < m; i++)
        std::cout << "===";
    std::cout << std::endl;
}

void compareCUDA2CPU(int nCols)
{
    thrust::host_vector<int> h_minors(nCols * (nCols - 1) / 2, -99);
    thrust::device_vector<int> d_minors(nCols * (nCols - 1) / 2, -99);
    thrust::host_vector<int> h_A(2 * nCols, -99);
    thrust::device_vector<int> d_A(2 * nCols, -99);
    for (int i = 0; i < 2 * nCols; ++i)
    {
        // int val = rand();
        int val = i;
        h_A[i] = val;
        d_A[i] = val;
    }

    minors2xN_device(nCols, nCols - 1, &d_minors, &d_A, nCols);
    cudaDeviceSynchronize();
    minors2xN_cpu(&h_minors, &h_A, nCols);

    printf("Host/CPU minors:\n");
    print_matrix(h_minors, 1, h_minors.size());
    printf("Device/GPU minors:\n");
    print_matrix(d_minors, 1, d_minors.size());
}

int main()
{
    int nCols = 5;
    compareCUDA2CPU(nCols);
    return 0;
}